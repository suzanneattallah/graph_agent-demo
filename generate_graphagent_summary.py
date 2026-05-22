"""Génère un PDF de synthèse du projet GraphAgent."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUTPUT = r"C:\Projet\graphagent_summary.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

TEAL   = colors.HexColor("#0d9488")
VIOLET = colors.HexColor("#7c3aed")
AMBER  = colors.HexColor("#d97706")
DARK   = colors.HexColor("#1e293b")
LIGHT  = colors.HexColor("#f1f5f9")
GREY   = colors.HexColor("#64748b")

def S(name, **kw):
    base = styles[name]
    return ParagraphStyle(name + "_custom", parent=base, **kw)

title_style    = S("Title",    textColor=TEAL,   fontSize=22, spaceAfter=6)
h1_style       = S("Heading1", textColor=TEAL,   fontSize=14, spaceBefore=14, spaceAfter=4, borderPad=2)
h2_style       = S("Heading2", textColor=VIOLET, fontSize=11, spaceBefore=10, spaceAfter=3)
body_style     = S("Normal",   textColor=DARK,   fontSize=9,  leading=14, alignment=TA_JUSTIFY)
code_style     = S("Code",     textColor=colors.HexColor("#0f4f4f"),
                   backColor=LIGHT, fontSize=8, leading=12, fontName="Courier",
                   leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)
caption_style  = S("Normal",   textColor=GREY,   fontSize=8,  leading=11, fontName="Helvetica-Oblique")
bullet_style   = S("Normal",   textColor=DARK,   fontSize=9,  leading=13, leftIndent=12)

def HR(): return HRFlowable(width="100%", thickness=0.5, color=TEAL, spaceAfter=6)
def SP(h=6): return Spacer(1, h)
def P(text, style=None): return Paragraph(text, style or body_style)
def H1(text): return Paragraph(text, h1_style)
def H2(text): return Paragraph(text, h2_style)
def Code(text): return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style)
def Bullet(items):
    return ListFlowable(
        [ListItem(P(i, bullet_style), bulletColor=TEAL, leftIndent=14) for i in items],
        bulletType="bullet", leftIndent=6,
    )

# ── Document ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm,  bottomMargin=2*cm,
)

story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story += [
    P("GraphAgent", title_style),
    P("Traçage en temps réel des mouvements d'un agent LLM dans un graphe de code",
      S("Normal", textColor=GREY, fontSize=11, spaceAfter=4)),
    HR(), SP(4),
]

# ── 1. Vue d'ensemble ─────────────────────────────────────────────────────────
story += [
    H1("1. Vue d'ensemble"),
    P("""GraphAgent est un agent ReAct (DSPy) qui navigue un graphe de connaissance Neo4j
représentant une base de code Java. Chaque nœud est une classe, méthode ou fichier ;
chaque arête est une relation (CALLS, EXTENDS, IMPORTS…). L'agent répond à des questions
d'analyse en explorant ce graphe via des outils Cypher, <b>sans jamais utiliser de
connaissances a priori</b>.
"""),
    SP(4),
]

# ── 2. Pipeline ───────────────────────────────────────────────────────────────
story += [
    H1("2. Pipeline complet"),
    SP(2),
]

pipeline_data = [
    ["Étape", "Outil / Script", "Sortie"],
    ["1. Parsing AST",      "python -m AST <projet> --lang .java",  "<projet>-ast.json"],
    ["2. Import Neo4j",     "python import_to_neo4j.py",            "Graphe Neo4j (~15k nœuds)"],
    ["3. Run agent",        "python -m graph_agent.run --model …",  "Réponse + traces MLflow"],
    ["4. Visualiseur",      "streamlit run graph_agent/visualizer.py", "Vue temps réel"],
]
t = Table(pipeline_data, colWidths=[4*cm, 7*cm, 5.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), TEAL),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
    ("GRID",         (0,0), (-1,-1), 0.3, GREY),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ("LEFTPADDING",  (0,0), (-1,-1), 6),
]))
story += [t, SP(6)]

# ── 3. Traçage temps réel ─────────────────────────────────────────────────────
story += [
    H1("3. Traçage en temps réel — nav_state.json"),
    P("""<b>Pourquoi ne pas utiliser MLflow directement ?</b><br/>
MLflow autolog (DSPy) bufferise les spans et ne les persiste qu'à la <i>fin du run</i>.
Interroger l'API MLflow pendant l'exécution retournerait des données incomplètes ou vides.
Le run peut durer plusieurs minutes — le visualiseur serait inutilisable.
"""),
    SP(4),
    P("""<b>Solution : un fichier JSON comme bus de communication synchrone.</b><br/>
À chaque appel d'outil, <code>agent.py</code> sérialise immédiatement l'état courant de
l'agent dans <code>nav_state.json</code>. Streamlit relit ce fichier toutes les secondes.
Latence effective : &lt; 50 ms.
"""),
    SP(6),
]

# Schéma textuel
schema_data = [
    ["agent.py  (LLM + tools)", "→  écrit", "nav_state.json", "←  relit  (1s)", "visualizer.py  (Streamlit)"],
]
ts = Table(schema_data, colWidths=[4.5*cm, 1.5*cm, 3.5*cm, 2.5*cm, 4.5*cm])
ts.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (0,0), colors.HexColor("#ccfbf1")),
    ("BACKGROUND",   (2,0), (2,0), colors.HexColor("#fef3c7")),
    ("BACKGROUND",   (4,0), (4,0), colors.HexColor("#ede9fe")),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
    ("ALIGN",        (0,0), (-1,-1), "CENTER"),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0), (-1,-1), 6),
    ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ("BOX",          (0,0), (-1,-1), 0.5, GREY),
    ("INNERGRID",    (0,0), (-1,-1), 0.3, GREY),
]))
story += [ts, SP(8)]

# ── 4. Structure de nav_state.json ────────────────────────────────────────────
story += [
    H2("Structure de nav_state.json"),
    Code("""{
  "current":    "impl_omsportalorderserviceimpl_generateorder",
  "visited":    ["demo_malldemoapplication", "controller_omsportalordercontroller"],
  "notes":      ["Controller delegates to portalOrderService.generateOrder()"],
  "moves":      [{"from": "...", "to": "...", "via": "tool_move_to", "timestamp": ...}],
  "tool_calls": [{"tool": "tool_read_neighbours", "node": "...", "args": {}, "timestamp": ...}]
}"""),
    SP(4),
    P("Ce fichier est <b>réécrit intégralement à chaque appel d'outil</b> (13 outils loggués). "
      "Streamlit lit la dernière version et met à jour le graphe et le feed lateral."),
    SP(6),
]

# ── 5. Outils de l'agent ──────────────────────────────────────────────────────
story += [
    H1("4. Outils de l'agent"),
    SP(2),
]

tools_data = [
    ["Outil", "Icône", "Rôle"],
    ["tool_read_node",          "🔍", "Lit toutes les propriétés du nœud courant"],
    ["tool_read_neighbours",    "🕸️", "Liste les voisins (filtre relation optionnel)"],
    ["tool_read_outgoing",      "➡️", "Nœuds vers lesquels pointe le nœud courant"],
    ["tool_read_incoming",      "⬅️", "Nœuds qui pointent vers le nœud courant"],
    ["tool_move_to",            "🔀", "Déplace l'agent sur un autre nœud"],
    ["tool_read_source_code",   "📄", "Lit le fichier Java source associé"],
    ["tool_search_node",        "🔎", "Recherche par nom (insensible à la casse)"],
    ["tool_find_path",          "🗺️", "Plus court chemin vers un nœud cible"],
    ["tool_get_call_chain",     "⛓️", "Chaîne d'appels sortants (5 hops)"],
    ["tool_get_callers",        "📞", "Qui appelle ce nœud (3 hops upstream)"],
    ["tool_get_parent_class",   "👆", "Classe parente d'une méthode"],
    ["tool_add_note",           "📝", "Sauvegarde une observation en mémoire"],
    ["tool_history",            "📋", "Retourne position, visités et notes"],
]
tt = Table(tools_data, colWidths=[5.5*cm, 1.2*cm, 9.8*cm])
tt.setStyle(TableStyle([
    ("BACKGROUND",   (0,0), (-1,0), VIOLET),
    ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
    ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",     (0,0), (-1,-1), 8),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
    ("GRID",         (0,0), (-1,-1), 0.3, GREY),
    ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",   (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0), (-1,-1), 3),
    ("LEFTPADDING",  (0,0), (-1,-1), 6),
]))
story += [tt, SP(6)]

# ── 6. Visualiseur Streamlit ──────────────────────────────────────────────────
story += [
    H1("5. Visualiseur Streamlit"),
    P("""Le visualiseur charge uniquement un <b>sous-graphe dynamique</b> depuis Neo4j :
nœud courant + nœuds visités + leurs voisins directs (1 hop). Sur un projet de 15 000 nœuds
comme <i>mall</i>, cela ramène l'affichage à ~50–200 nœuds, rendant la page fluide.
"""),
    SP(4),
    H2("Légende du graphe"),
    Bullet([
        "<b>Hexagone blanc (teal)</b> — nœud courant de l'agent",
        "<b>Cercle violet</b> — nœud déjà visité",
        "<b>Flèche teal pleine</b> — arête existante dans Neo4j, traversée par l'agent",
        "<b>Flèche ambrée pointillée</b> — saut de navigation virtuel "
        "(pas d'arête directe dans le graphe), labelisé avec le nom de l'outil utilisé",
        "<b>Cercle gris</b> — voisin non visité (contexte)",
    ]),
    SP(6),
    H2("Feed latéral — Tool calls"),
    P("""La sidebar affiche en temps réel le flux des appels d'outils (30 derniers),
avec icône, nom court, arguments principaux et le nœud sur lequel l'appel a eu lieu.
Cela permet de suivre le <i>raisonnement</i> de l'agent pas à pas sans attendre la réponse finale.
"""),
    SP(6),
]

# ── 7. Modèles ────────────────────────────────────────────────────────────────
story += [
    H1("6. Modèles testés"),
    Bullet([
        "<b>qwen2.5-coder:7b</b> (Ollama local) — modèle par défaut, limité sur les grands graphes",
        "<b>qwen3.6-27b</b> (endpoint Mac interne) — recommandé pour l'exploration complète",
        "<b>qwen3.6-35b-a3b</b> (endpoint Mac interne) — plus puissant, MoE 35B",
    ]),
    SP(4),
    P("Connexion à un endpoint OpenAI-compatible via <code>--api-base</code> :"),
    Code("python -m graph_agent.run\n"
         "  --model qwen3.6-27b\n"
         "  --api-base http://CHLASLITASSAPR1.lan.la.sqli.com:8080/v1\n"
         '  --question "..."'),
    SP(6),
]

# ── 8. Projet testé ───────────────────────────────────────────────────────────
story += [
    H1("7. Projet analysé — macrozheng/mall"),
    P("""Plateforme e-commerce Spring Boot complète (~100k LOC Java).
Parsée avec l'outil AST maison (tree-sitter) : <b>14 996 nœuds</b> et <b>28 129 arêtes</b>
(14 143 méthodes, 9 254 appels, 3 743 imports, 754 contains, 235 extends/implements).
"""),
    SP(4),
    Code("python -m AST C:\\Projet\\mall --output C:\\Projet\\mall-ast.json --lang .java --stats"),
    SP(8),
    HR(),
    P("GraphAgent — Projet interne SQLI · Mai 2026", caption_style),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"✅  PDF généré : {OUTPUT}")
