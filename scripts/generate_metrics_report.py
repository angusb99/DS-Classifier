"""Render a PDF model metrics report from the pipeline's prediction outputs.

Reads the CSVs written by training_functions (no model or retraining needed):
    data/output/{period}_{state}_training_prediction_validation_output.csv   (required)
    data/output/{period}_{state}_{pool}_predictions.csv                      (optional, candidate section)
    a benchmark JSON from scripts/benchmark_pipeline.py                       (optional, --benchmark)

and writes data/output/reports/{period}_{state}_metrics_report.pdf. With --adhoc-output, the CSVs are read from
and the report written to that folder instead, matching run_pipeline.py --adhoc-output.

Example:
    .venv\\Scripts\\python scripts\\generate_metrics_report.py --period a25 --state SA --pool all
"""
import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score, cohen_kappa_score, confusion_matrix, log_loss, matthews_corrcoef,
    precision_recall_fscore_support
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
NON_PROBABILITY_COLUMNS = {'predicted_class', 'max_prob', 'ARID', 'true_class', 'original_class'}
OTHER_DWELLING_CODES = {'x35', 'x91', 'x92', 'x93', 'x94'}

# Light-mode tokens from the reference data-viz palette (print is always light)
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']   # first three categorical slots - validated all-pairs
SEQUENTIAL_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def validation_metrics(val, class_names):
    y_true, y_pred = val['true_class'], val['predicted_class']
    probs = val[class_names].to_numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)  # CSV rounding leaves rows a hair off 1
    true_idx = pd.Categorical(y_true, categories=class_names).codes
    top2 = np.argsort(probs, axis=1)[:, -2:]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    summary = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Top-2 accuracy': float(np.mean((top2 == true_idx[:, None]).any(axis=1))),
        'Macro F1': f1.mean(),
        'Weighted F1': np.average(f1, weights=support),
        "Cohen's kappa": cohen_kappa_score(y_true, y_pred),
        'Matthews corr.': matthews_corrcoef(y_true, y_pred),
        'Log loss': log_loss(true_idx, probs, labels=range(len(class_names))),
    }
    per_class = pd.DataFrame(
        {'Support': support, 'Precision': precision, 'Recall': recall, 'F1': f1}, index=class_names
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return summary, per_class, cm


def top_misclassifications(cm, class_names, n=10):
    off_diagonal = cm * (1 - np.eye(len(cm), dtype=cm.dtype))
    rows = [
        (class_names[i], class_names[j], int(off_diagonal[i, j]), off_diagonal[i, j] / cm[i].sum())
        for i, j in zip(*np.nonzero(off_diagonal))
    ]
    return sorted(rows, key=lambda r: r[2], reverse=True)[:n]


def calibration_table(val, bins=10):
    correct = (val['true_class'] == val['predicted_class']).to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(val['max_prob'], edges[1:-1]), 0, bins - 1)
    table = (
        pd.DataFrame({'bucket': bucket, 'confidence': val['max_prob'], 'correct': correct})
        .groupby('bucket')
        .agg(rows=('correct', 'size'), confidence=('confidence', 'mean'), accuracy=('correct', 'mean'))
        .reindex(range(bins))
    )
    table.insert(0, 'range', [f"{edges[i]:.1f} - {edges[i + 1]:.1f}" for i in range(bins)])
    table = table.dropna(subset=['rows'])
    ece = float(np.nansum(table['rows'] * (table['confidence'] - table['accuracy']).abs()) / len(val))
    return table, ece


def candidate_summary(cand, class_names):
    original = cand['original_class']
    if 'OTHER' in class_names:
        # Model was trained with the minor codes collapsed, so compare like with like
        original = original.where(~original.isin(OTHER_DWELLING_CODES), 'OTHER')
    has_original = original.notna()
    distribution = pd.DataFrame({
        'Predicted': cand['predicted_class'].value_counts(normalize=True),
        'Original': original[has_original].value_counts(normalize=True),
    }).reindex(class_names).fillna(0)
    confidence_bands = pd.cut(
        cand['max_prob'], [0, 0.5, 0.7, 0.9, 1.0001],
        labels=['< 0.5', '0.5 - 0.7', '0.7 - 0.9', '>= 0.9'], right=False
    ).value_counts(sort=False)
    return {
        'rows': len(cand),
        'with_original': int(has_original.sum()),
        'agreement': float((cand.loc[has_original, 'predicted_class'] == original[has_original]).mean())
        if has_original.any() else float('nan'),
        'median_confidence': float(cand['max_prob'].median()),
        'distribution': distribution,
        'confidence_bands': confidence_bands,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style_axes(ax, grid_axis='y'):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(width_mm, height_mm):
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _to_image(fig, width_mm, height_mm):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=width_mm * mm, height=height_mm * mm, kind='proportional')


def confusion_matrix_chart(cm, class_names):
    size = 150
    fig, ax = _figure(size, size * 0.85)
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list('seq_blue', [SURFACE] + SEQUENTIAL_BLUE)
    # origin='lower' puts the first class at the bottom, so the diagonal runs bottom-left to top-right
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1, origin='lower')
    for i in range(len(cm)):
        for j in range(len(cm)):
            ink = SURFACE if row_share[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i + 0.12, f"{row_share[i, j]:.0%}", ha='center', va='center', fontsize=8, color=ink,
                    fontweight='bold' if i == j else 'normal')
            ax.text(j, i - 0.22, f"{cm[i, j]:,}", ha='center', va='center', fontsize=6, color=ink)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel('Predicted class', color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel('True class', color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(cm)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cm)), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)
    return _to_image(fig, size, size * 0.85)


def per_class_chart(per_class):
    width, height = 170, 70
    fig, ax = _figure(width, height)
    x = np.arange(len(per_class))
    bar_width = 0.26
    for k, metric in enumerate(['Precision', 'Recall', 'F1']):
        ax.bar(x + (k - 1) * bar_width, per_class[metric], bar_width, label=metric,
               color=SERIES[k], edgecolor=SURFACE, linewidth=1)
    _style_axes(ax)
    ax.set_xticks(x, per_class.index)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
              fontsize=8, labelcolor=TEXT_SECONDARY)
    return _to_image(fig, width, height)


def calibration_chart(table):
    size = 85
    fig, ax = _figure(size, size)
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, linestyle='--', label='Perfect calibration')
    ax.plot(table['confidence'], table['accuracy'], color=SERIES[0], linewidth=2, marker='o',
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label='Model')
    _style_axes(ax, grid_axis='both')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(BASELINE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted confidence', color=TEXT_SECONDARY, fontsize=8)
    ax.set_ylabel('Observed accuracy', color=TEXT_SECONDARY, fontsize=8)
    ax.legend(loc='upper left', frameon=False, fontsize=7, labelcolor=TEXT_SECONDARY)
    return _to_image(fig, size, size)


def distribution_chart(distribution):
    width, height = 170, 65
    fig, ax = _figure(width, height)
    x = np.arange(len(distribution))
    bar_width = 0.38
    for k, column in enumerate(['Predicted', 'Original']):
        ax.bar(x + (k - 0.5) * bar_width, distribution[column], bar_width, label={'Predicted': 'Predicted class', 'Original': 'Existing (imputed / admin) class'}[column],
               color=SERIES[k], edgecolor=SURFACE, linewidth=1)
    _style_axes(ax)
    ax.set_xticks(x, distribution.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False,
              fontsize=8, labelcolor=TEXT_SECONDARY)
    return _to_image(fig, width, height)


# ---------------------------------------------------------------------------
# PDF layout
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', parent=base['Title'], fontSize=20, leading=24,
                                textColor=colors.HexColor(TEXT_PRIMARY), alignment=0, spaceAfter=2),
        'subtitle': ParagraphStyle('subtitle', parent=base['Normal'], fontSize=9.5,
                                   textColor=colors.HexColor(TEXT_SECONDARY), spaceAfter=10),
        'h1': ParagraphStyle('h1', parent=base['Heading2'], fontSize=13, leading=16,
                             textColor=colors.HexColor(TEXT_PRIMARY), spaceBefore=10, spaceAfter=4),
        'body': ParagraphStyle('body', parent=base['Normal'], fontSize=9, leading=12.5,
                               textColor=colors.HexColor(TEXT_SECONDARY), spaceAfter=6),
        'kpi_value': ParagraphStyle('kpi_value', parent=base['Normal'], fontSize=16, leading=19,
                                    textColor=colors.HexColor(TEXT_PRIMARY), alignment=TA_CENTER,
                                    fontName='Helvetica-Bold'),
        'kpi_label': ParagraphStyle('kpi_label', parent=base['Normal'], fontSize=7.5, leading=9,
                                    textColor=colors.HexColor(TEXT_MUTED), alignment=TA_CENTER),
        'cell': ParagraphStyle('cell', parent=base['Normal'], fontSize=8, leading=10,
                               textColor=colors.HexColor(TEXT_PRIMARY)),
    }


def kpi_row(items, styles):
    cells = [[Paragraph(value, styles['kpi_value']) for _, value in items],
             [Paragraph(label, styles['kpi_label']) for label, _ in items]]
    table = Table(cells, colWidths=[CONTENT_WIDTH / len(items)] * len(items))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f3f2ee')),
        ('LINEAFTER', (0, 0), (-2, -1), 2, colors.white),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
    ]))
    return table


def data_table(header, rows, col_widths, numeric_from=1):
    table = Table([header] + rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8),
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(TEXT_SECONDARY)),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(TEXT_PRIMARY)),
        ('ALIGN', (numeric_from, 0), (-1, -1), 'RIGHT'),
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, colors.HexColor(BASELINE)),
        ('LINEBELOW', (0, 1), (-1, -1), 0.3, colors.HexColor(GRIDLINE)),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return table


def page_decorations(title):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor(TEXT_MUTED))
        canvas.drawString(MARGIN, 10 * mm, title)
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def build_report(args):
    output_dir = OUTPUT_DIR / args.adhoc_output if args.adhoc_output else OUTPUT_DIR
    val_path = output_dir / f"{args.period}_{args.state}_training_prediction_validation_output.csv"
    cand_path = output_dir / f"{args.period}_{args.state}_{args.pool}_predictions.csv"
    if not val_path.exists():
        raise SystemExit(f"Validation predictions not found: {val_path}\nRun the pipeline first.")

    val = pd.read_csv(val_path)
    class_names = [c for c in val.columns if c not in NON_PROBABILITY_COLUMNS]
    summary, per_class, cm = validation_metrics(val, class_names)
    calibration, ece = calibration_table(val)
    misclassified = top_misclassifications(cm, class_names)

    styles = build_styles()
    title = f"Dwelling structure classifier - {args.state} {args.period}"
    story = [
        Paragraph(title, styles['title']),
        Paragraph(
            f"Model metrics report &middot; generated {datetime.now():%d %b %Y %H:%M} &middot; "
            f"{len(val):,} validation records &middot; {len(class_names)} classes", styles['subtitle']
        ),
        kpi_row([(k, f"{summary[k]:.1%}") for k in ['Accuracy', 'Top-2 accuracy', 'Macro F1', 'Weighted F1']], styles),
        Spacer(1, 4),
        kpi_row([(k, f"{summary[k]:.3f}") for k in ["Cohen's kappa", 'Matthews corr.', 'Log loss']]
                + [('Calibration error (ECE)', f"{ece:.3f}")], styles),
        Paragraph("Validation performance", styles['h1']),
        Paragraph(
            "Metrics are computed on the 10% stratified hold-out set the model did not train on. Macro scores "
            "weight every class equally, so they expose weak minority classes that accuracy hides.",
            styles['body']
        ),
        Paragraph("Per-class metrics", styles['h1']),
        per_class_chart(per_class),
        data_table(
            ['Class', 'Support', 'Precision', 'Recall', 'F1'],
            [[name, f"{int(r.Support):,}", f"{r.Precision:.3f}", f"{r.Recall:.3f}", f"{r.F1:.3f}"]
             for name, r in per_class.iterrows()],
            [CONTENT_WIDTH * w for w in (0.28, 0.18, 0.18, 0.18, 0.18)]
        ),
        PageBreak(),
        Paragraph("Confusion matrix", styles['h1']),
        Paragraph("Each row is a true class; cells show the share of that class predicted as each column, "
                  "with the record count beneath.", styles['body']),
        confusion_matrix_chart(cm, class_names),
        KeepTogether([
            Paragraph("Most common misclassifications", styles['h1']),
            data_table(
                ['True class', 'Predicted as', 'Records', 'Share of true class'],
                [[t, p, f"{n:,}", f"{share:.1%}"] for t, p, n, share in misclassified],
                [CONTENT_WIDTH * w for w in (0.25, 0.25, 0.25, 0.25)], numeric_from=2
            ),
        ]),
        PageBreak(),
        Paragraph("Confidence and calibration", styles['h1']),
        Paragraph(
            f"A well calibrated model's confidence matches how often it is right; points below the diagonal "
            f"are over-confident. Expected calibration error is {ece:.3f} (0 is perfect).", styles['body']
        ),
        Table([[
            calibration_chart(calibration),
            data_table(
                ['Confidence', 'Records', 'Mean conf.', 'Accuracy'],
                [[r.range, f"{int(r.rows):,}", f"{r.confidence:.3f}", f"{r.accuracy:.1%}"]
                 for r in calibration.itertuples()],
                [22 * mm, 18 * mm, 18 * mm, 18 * mm]
            )
        ]], colWidths=[CONTENT_WIDTH * 0.52, CONTENT_WIDTH * 0.48], style=[('VALIGN', (0, 0), (-1, -1), 'TOP')]),
    ]

    if cand_path.exists():
        cand = candidate_summary(pd.read_csv(cand_path), class_names)
        story += [
            Paragraph(f"Candidate predictions ({args.pool} pool)", styles['h1']),
            kpi_row([
                ('Candidates scored', f"{cand['rows']:,}"),
                ('Median confidence', f"{cand['median_confidence']:.1%}"),
                ('With existing class', f"{cand['with_original']:,}"),
                ('Agreement with existing', f"{cand['agreement']:.1%}" if cand['with_original'] else "n/a"),
            ], styles),
            Spacer(1, 6),
            Paragraph("Share of candidates by predicted class, against the class currently recorded "
                      "(imputed or admin sourced) where one exists.", styles['body']),
            distribution_chart(cand['distribution']),
            data_table(
                ['Confidence band', 'Candidates', 'Share'],
                [[band, f"{n:,}", f"{n / cand['rows']:.1%}"] for band, n in cand['confidence_bands'].items()],
                [CONTENT_WIDTH * w for w in (0.4, 0.3, 0.3)]
            ),
        ]

    if args.benchmark:
        bench = json.loads(Path(args.benchmark).read_text())
        cfg = bench['config']
        story += [
            PageBreak(),
            Paragraph("Run performance", styles['h1']),
            Paragraph(
                f"From {Path(args.benchmark).name}: {cfg['records']:,} records, {bench['status']}, "
                f"total {bench.get('total_seconds', float('nan')) / 60:.1f} min, overall peak memory "
                f"{bench['overall_peak_rss'] / 1024 ** 3:.2f} GB on {bench['system']['total_ram_gb']:.0f} GB RAM.",
                styles['body']
            ),
            data_table(
                ['Stage', 'Time (s)', 'Peak RSS (GB)', 'Peak increase (GB)'],
                [["   " * s['depth'] + s['name'], f"{s.get('seconds', float('nan')):.1f}",
                  f"{s['peak_rss'] / 1024 ** 3:.2f}", f"{(s['peak_rss'] - s['start_rss']) / 1024 ** 3:.2f}"]
                 for s in bench['stages']],
                [CONTENT_WIDTH * w for w in (0.46, 0.18, 0.18, 0.18)]
            ),
        ]

    report_path = output_dir / "reports" / f"{args.period}_{args.state}_metrics_report.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(report_path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
        bottomMargin=MARGIN, title=title, author="DS Classifier pipeline"
    )
    decorate = page_decorations(title)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"Report written to {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", default="a25")
    parser.add_argument("--state", default="SA")
    parser.add_argument("--pool", default="all", help="candidate pool whose predictions to summarise")
    parser.add_argument("--adhoc-output", help="ad hoc output folder the pipeline was run with")
    parser.add_argument("--benchmark", help="optional benchmark JSON from scripts/benchmark_pipeline.py")
    build_report(parser.parse_args())
