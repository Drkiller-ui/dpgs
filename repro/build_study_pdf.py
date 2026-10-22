"""Build the study document from its Markdown source using bundled ReportLab."""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted

ROOT = Path(__file__).resolve().parents[1]
font = 'StudyChinese'
pdfmetrics.registerFont(TTFont(font, 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
styles = {
    'body': ParagraphStyle('Body', fontName=font, fontSize=10.2, leading=17.5,
                           spaceAfter=8, wordWrap='CJK', textColor=colors.HexColor('#202428')),
    'h1': ParagraphStyle('Title', fontName=font, fontSize=22, leading=31, spaceAfter=19,
                         wordWrap='CJK', textColor=colors.black),
    'h2': ParagraphStyle('Section', fontName=font, fontSize=15, leading=23, spaceBefore=15,
                         spaceAfter=10, keepWithNext=True, wordWrap='CJK', textColor=colors.black),
    'h3': ParagraphStyle('Subsection', fontName=font, fontSize=11.5, leading=19, spaceBefore=9,
                         spaceAfter=7, keepWithNext=True, wordWrap='CJK', textColor=colors.black),
    'cell': ParagraphStyle('Cell', fontName=font, fontSize=8.3, leading=13.2, wordWrap='CJK'),
    'code': ParagraphStyle('Code', fontName=font, fontSize=8.0, leading=14, spaceAfter=8),
}


def markup(s):
    s = escape(s)
    s = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#164A72">\1</link>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    return s


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(font, 8)
    canvas.setFillColor(colors.HexColor('#707070'))
    canvas.drawString(46, 28, '四篇三维重建论文学习笔记')
    canvas.drawRightString(A4[0] - 46, 28, str(doc.page))
    canvas.restoreState()


def main():
    source = ROOT / 'docs/四篇论文关系学习笔记.md'
    output = ROOT / 'docs/四篇论文关系学习笔记.pdf'
    lines = source.read_text(encoding='utf-8').splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            story.append(Preformatted('\n'.join(code), styles['code']))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                fields = [x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[- :]+', x) for x in fields):
                    rows.append([Paragraph(markup(x), styles['cell']) for x in fields])
                i += 1
            widths = [101, 102, 145, 155] if len(rows[0]) == 4 else [120, 155, 228]
            tab = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5EDF3')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F9FA')]),
                ('GRID', (0, 0), (-1, -1), .5, colors.HexColor('#D9D9D9')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
            ]))
            story.extend([tab, Spacer(1, 12)])
            continue
        if line.startswith('# '):
            story.append(Paragraph(markup(line[2:]), styles['h1']))
        elif line.startswith('## '):
            story.append(Paragraph(markup(line[3:]), styles['h2']))
        elif line.startswith('### '):
            story.append(Paragraph(markup(line[4:]), styles['h3']))
        else:
            story.append(Paragraph(markup(line), styles['body']))
        i += 1
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=46, leftMargin=46,
                      topMargin=43, bottomMargin=46, title='四篇三维重建论文的关系与学习路线',
                      author='').build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == '__main__':
    main()
