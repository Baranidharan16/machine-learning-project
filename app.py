from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os
import io
import base64
import csv
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def parse_transactions(filepath):
    """Parse CSV → list of item-lists, supporting multiple formats."""
    transactions = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty file")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(k in header for k in ('items', 'item', 'products', 'product', 'transaction'))

    if has_header:
        # Find the items column (everything after an ID col)
        item_col = None
        for i, h in enumerate(header):
            if h in ('items', 'item', 'products', 'product', 'description'):
                item_col = i
                break
        if item_col is None:
            item_col = 1 if len(header) > 1 else 0

        for row in rows[1:]:
            if not row:
                continue
            cell = row[item_col].strip()
            if cell:
                items = [i.strip() for i in cell.split(',') if i.strip()]
                if items:
                    transactions.append(items)
    else:
        for row in rows:
            items = [c.strip() for c in row if c.strip()]
            if items:
                transactions.append(items)

    return transactions


def run_apriori(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_items = apriori(df, min_support=min_support, use_colnames=True)
    if frequent_items.empty:
        return None, None, df, te.columns_

    frequent_items['length'] = frequent_items['itemsets'].apply(len)
    frequent_items = frequent_items.sort_values('support', ascending=False)

    rules = association_rules(frequent_items, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False)

    return frequent_items, rules, df, te.columns_


# ── chart generators ──────────────────────────────────────────────────────────

DARK_BG   = '#0d0f14'
CARD_BG   = '#151820'
ACCENT1   = '#00e5ff'
ACCENT2   = '#ff6b6b'
ACCENT3   = '#7fff7f'
TEXT_CLR  = '#e8ecf0'

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': TEXT_CLR,
    'axes.labelcolor': TEXT_CLR,
    'xtick.color': TEXT_CLR,
    'ytick.color': TEXT_CLR,
})


def chart_top_items(transactions):
    all_items = [item for t in transactions for item in t]
    counts = Counter(all_items).most_common(15)
    labels, vals = zip(*counts)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(vals)))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], edgecolor='none', height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=9, color=ACCENT1)
    ax.set_xlabel('Frequency', color=TEXT_CLR, fontsize=10)
    ax.set_title('Top 15 Most Frequent Items', color=ACCENT1, fontsize=13, pad=15, fontweight='bold')
    ax.spines[:].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, max(vals)*1.12)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_support_confidence(rules_df):
    if rules_df is None or rules_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    sc = ax.scatter(rules_df['support'], rules_df['confidence'],
                    c=rules_df['lift'], cmap='plasma',
                    s=rules_df['lift']*30, alpha=0.85, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Lift', color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
    ax.set_xlabel('Support', color=TEXT_CLR, fontsize=11)
    ax.set_ylabel('Confidence', color=TEXT_CLR, fontsize=11)
    ax.set_title('Support vs Confidence (sized/colored by Lift)', color=ACCENT1,
                 fontsize=13, pad=15, fontweight='bold')
    ax.spines[:].set_color('#2a2f3d')
    ax.tick_params(labelsize=9)
    ax.grid(True, color='#1e2330', linewidth=0.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_heatmap(rules_df, top_n=12):
    if rules_df is None or rules_df.empty:
        return None
    subset = rules_df.head(top_n).copy()
    subset['antecedents_str'] = subset['antecedents'].apply(lambda x: ', '.join(list(x)))
    subset['consequents_str'] = subset['consequents'].apply(lambda x: ', '.join(list(x)))

    pivot_data = subset.pivot_table(
        index='antecedents_str', columns='consequents_str',
        values='confidence', aggfunc='max', fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, pivot_data.shape[1]*1.2),
                                    max(5, pivot_data.shape[0]*0.8)),
                           facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(pivot_data, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                linewidths=0.3, linecolor='#1e2330',
                cbar_kws={'shrink': 0.8})
    ax.set_title('Confidence Heatmap: Antecedents → Consequents',
                 color=ACCENT1, fontsize=13, pad=15, fontweight='bold')
    ax.set_xlabel('Consequents', color=TEXT_CLR)
    ax.set_ylabel('Antecedents', color=TEXT_CLR)
    ax.tick_params(labelsize=8, colors=TEXT_CLR)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_lift_bar(rules_df, top_n=10):
    if rules_df is None or rules_df.empty:
        return None
    subset = rules_df.head(top_n).copy()
    subset['rule'] = (subset['antecedents'].apply(lambda x: ', '.join(list(x))) +
                      ' → ' +
                      subset['consequents'].apply(lambda x: ', '.join(list(x))))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    colors = plt.cm.autumn_r(np.linspace(0, 0.8, len(subset)))
    bars = ax.barh(subset['rule'][::-1], subset['lift'][::-1],
                   color=colors, edgecolor='none', height=0.55)
    ax.axvline(1.0, color='#ffffff44', linewidth=1, linestyle='--')
    for bar, v in zip(bars, subset['lift'][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}', va='center', fontsize=8.5, color=ACCENT3)
    ax.set_xlabel('Lift', color=TEXT_CLR, fontsize=11)
    ax.set_title('Top Rules by Lift Score', color=ACCENT1, fontsize=13, pad=15, fontweight='bold')
    ax.spines[:].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        min_support    = float(request.form.get('min_support', 0.1))
        min_confidence = float(request.form.get('min_confidence', 0.5))
        min_lift       = float(request.form.get('min_lift', 1.0))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.csv')
        f.save(filepath)

        transactions = parse_transactions(filepath)
        if len(transactions) < 2:
            return jsonify({'error': 'Need at least 2 transactions'}), 400

        frequent_items, rules_df, _, all_items = run_apriori(
            transactions, min_support, min_confidence, min_lift)

        # ── summary stats ──────────────────────────────────────────────────────
        all_flat = [i for t in transactions for i in t]
        item_counts = Counter(all_flat).most_common(5)
        summary = {
            'total_transactions': len(transactions),
            'unique_items': len(set(all_flat)),
            'avg_basket_size': round(np.mean([len(t) for t in transactions]), 2),
            'top_item': item_counts[0][0] if item_counts else '-',
            'frequent_itemsets': 0,
            'rules_found': 0,
        }

        frequent_list, rules_list = [], []

        if frequent_items is not None and not frequent_items.empty:
            summary['frequent_itemsets'] = len(frequent_items)
            frequent_list = [
                {
                    'items': ', '.join(list(row['itemsets'])),
                    'support': round(row['support'], 4),
                    'length': int(row['length']),
                }
                for _, row in frequent_items.iterrows()
            ]

        if rules_df is not None and not rules_df.empty:
            summary['rules_found'] = len(rules_df)
            rules_list = [
                {
                    'antecedents': ', '.join(list(row['antecedents'])),
                    'consequents': ', '.join(list(row['consequents'])),
                    'support':    round(row['support'], 4),
                    'confidence': round(row['confidence'], 4),
                    'lift':       round(row['lift'], 4),
                }
                for _, row in rules_df.head(50).iterrows()
            ]

        # ── charts ─────────────────────────────────────────────────────────────
        charts = {
            'top_items':          chart_top_items(transactions),
            'support_confidence': chart_support_confidence(rules_df),
            'heatmap':            chart_heatmap(rules_df),
            'lift_bar':           chart_lift_bar(rules_df),
        }

        # ── save result CSV ────────────────────────────────────────────────────
        if rules_list:
            out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'rules.csv')
            pd.DataFrame(rules_list).to_csv(out_path, index=False)

        return jsonify({
            'success': True,
            'summary': summary,
            'frequent_itemsets': frequent_list[:30],
            'rules': rules_list,
            'charts': charts,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_rules')
def download_rules():
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'rules.csv')
    if os.path.exists(out_path):
        return send_file(out_path, mimetype='text/csv',
                         as_attachment=True, download_name='association_rules.csv')
    return jsonify({'error': 'No results yet'}), 404


@app.route('/api/sample_dataset')
def sample_dataset():
    """Return a sample dataset for download / demo."""
    sample = (
        "TransactionID,Items\n"
        "1,Bread,Milk\n"
        "2,Bread,Butter\n"
        "3,Milk,Butter\n"
        "4,Bread,Milk,Butter\n"
        "5,Bread,Milk,Eggs\n"
        "6,Milk,Eggs\n"
        "7,Butter,Eggs\n"
        "8,Bread,Butter,Eggs\n"
        "9,Bread,Milk,Butter,Eggs\n"
        "10,Milk,Butter,Eggs\n"
        "11,Bread,Jam\n"
        "12,Bread,Milk,Jam\n"
        "13,Butter,Jam\n"
        "14,Bread,Butter,Jam\n"
        "15,Milk,Jam\n"
        "16,Cheese,Bread\n"
        "17,Cheese,Milk\n"
        "18,Cheese,Butter\n"
        "19,Cheese,Eggs\n"
        "20,Cheese,Bread,Milk\n"
    )
    buf = io.BytesIO(sample.encode())
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name='sample_transactions.csv')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
