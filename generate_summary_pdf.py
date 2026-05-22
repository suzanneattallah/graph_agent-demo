from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from PIL import Image as PILImage
from pathlib import Path

OUT = Path(r"C:\Projet\prototype_summary.pdf")
IMG = Path(r"C:\Projet")
W, H = A4
M = 1.6 * cm

styles = getSampleStyleSheet()
TITLE   = ParagraphStyle("T",  parent=styles["Title"],   fontSize=22, spaceAfter=6,  textColor=colors.HexColor("#0f172a"), alignment=TA_CENTER)
SUBTITLE= ParagraphStyle("ST", parent=styles["Normal"],  fontSize=10, spaceAfter=14, textColor=colors.HexColor("#475569"), alignment=TA_CENTER)
H1      = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=14, spaceBefore=18,spaceAfter=6, textColor=colors.HexColor("#0f172a"))
H2      = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=11, spaceBefore=10,spaceAfter=4, textColor=colors.HexColor("#1e40af"))
BODY    = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9,  spaceAfter=5,  leading=13, textColor=colors.HexColor("#1e293b"), alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle("C",  parent=styles["Normal"],  fontSize=7.5,spaceAfter=8,  textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, fontName="Helvetica-Oblique")
BULLET  = ParagraphStyle("BU", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
TEAL    = colors.HexColor("#0d9488")
SLATE   = colors.HexColor("#e2e8f0")

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6, spaceBefore=6)

def img_block(filename, width_cm, caption=None):
    path = IMG / filename
    if not path.exists():
        return [Spacer(1, 0.5*cm)]
    pil = PILImage.open(str(path))
    pw, ph = pil.size
    w = width_cm * cm
    h = w * ph / pw
    elems = [RLImage(str(path), width=w, height=h)]
    if caption:
        elems.append(Paragraph(caption, CAPTION))
    return elems

def badge(n, title, color=TEAL):
    data = [[Paragraph(f"<font color='white'><b>Etape {n}</b></font>",
                       ParagraphStyle("b", fontSize=9, fontName="Helvetica-Bold")),
             Paragraph(f"<b>{title}</b>",
                       ParagraphStyle("t", fontSize=11, fontName="Helvetica-Bold",
                                      textColor=colors.HexColor("#0f172a")))]]
    t = Table(data, colWidths=[2*cm, W - M*2 - 2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), color),
        ("BACKGROUND", (1,0),(1,0), SLATE),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    return t

def tool_table(rows):
    data = [[Paragraph("<b>Outil / Composant</b>", BODY), Paragraph("<b>Role</b>", BODY)]] + \
           [[Paragraph(f"<b>{r[0]}</b>", BODY), Paragraph(r[1], BODY)] for r in rows]
    t = Table(data, colWidths=[4.2*cm, W - M*2 - 4.2*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0,0),(-1,-1), 0.3, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 7),
    ]))
    return t

def build():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M)
    s = []

    # Couverture
    s += [Spacer(1,1.2*cm),
          Paragraph("Agent de Navigation sur Graphe de Connaissance", TITLE),
          Paragraph("Prototype -- Analyse de codebase Java avec Neo4j, DSPy, MLflow et Streamlit", SUBTITLE),
          hr(),
          Paragraph(
            "Ce prototype analyse statiquement un projet Java (Spring PetClinic) et le transforme "
            "en graphe de connaissance stocke dans Neo4j. Un agent LLM (DSPy ReAct) navigue "
            "ensuite ce graphe de maniere autonome pour repondre a des questions architecturales "
            "complexes. Chaque execution est tracee dans MLflow et visualisee en temps reel "
            "dans Streamlit.", BODY),
          Spacer(1,0.4*cm)]

    # Stack
    s += [Paragraph("Pile Technologique", H1),
          tool_table([
            ("Spring PetClinic",         "Application Java de reference (~5 000 lignes) -- la codebase analysee"),
            ("Intelligy IDE + Outil AST", "Analyse statique du code Java et generation d'un graphe JSON (noeuds + aretes)"),
            ("Neo4j (Podman)",            "Base de donnees graphe stockant l'AST : classes, methodes, fichiers et relations"),
            ("DSPy ReAct",                "Orchestration LLM -- boucle Reflechir / Agir (outil) / Observer, 13 outils Cypher"),
            ("Ollama qwen2.5-coder:7b",   "LLM local -- tout le raisonnement s'execute en local, sans dependance cloud"),
            ("MLflow 3.12",               "Suivi des params, metriques, artefacts et arbre complet des appels d'outils"),
            ("Streamlit",                 "Tableau de bord temps reel -- le graphe se met a jour chaque seconde"),
          ]),
          Spacer(1,0.3*cm)]

    # Etape 1
    s += [badge(1, "Extraction de l'AST Java"), Spacer(1,0.2*cm),
          Paragraph(
            "Le projet Spring PetClinic est passe dans l'outil d'analyse AST "
            "(<code>C:\\Projet\\graphify-7\\AST</code>). Chaque fichier <code>.java</code> est "
            "analyse statiquement pour produire <b>spring-petclinic-ast.json</b> contenant "
            "<b>155 noeuds</b> et <b>384 aretes</b> representant les classes, methodes, fichiers "
            "et leurs relations (CONTAINS, METHOD, CALLS, EXTENDS, IMPLEMENTS, IMPORTS).", BODY),
          Spacer(1,0.2*cm)]

    # Etape 2
    s += [badge(2, "Import dans Neo4j"), Spacer(1,0.2*cm),
          Paragraph(
            "Le JSON est importe dans Neo4j via <b>import_to_neo4j.py</b>. Le graphe contient "
            "<b>25 noeuds Class</b>, <b>58 noeuds Method</b>, <b>25 noeuds File</b> et "
            "<b>7 noeuds External</b>, visibles dans le navigateur Neo4j ci-dessous.", BODY),
          Spacer(1,0.15*cm)]
    s += img_block("neo4j_graph.png", 15.5,
                   "Fig. 1 -- Navigateur Neo4j : 115 noeuds et 100 relations (Class / Method / File / External)")
    s += [Spacer(1,0.25*cm)]

    # Etape 3
    s += [badge(3, "GraphAgent -- DSPy ReAct + Ollama"), Spacer(1,0.2*cm),
          Paragraph(
            "L'agent (<b>graph_agent/agent.py</b>) maintient un <code>AgentState</code> "
            "(noeud courant, liste des noeuds visites, notes) et navigue le graphe via "
            "13 outils pre-construits -- aucune requete Cypher manuelle requise. "
            "Le noeud de depart est detecte automatiquement depuis les mots-cles de la question "
            "(en Python pur, sans appel LLM) ce qui elimine un aller-retour LLM inutile.", BODY),
          Spacer(1,0.1*cm),
          Paragraph("<b>Outils disponibles :</b>", H2),
          Paragraph("- <b>search_node(name)</b> -- recherche un noeud par sous-chaine de son label", BULLET),
          Paragraph("- <b>move_to(id)</b> -- deplace l'agent et met a jour nav_state.json instantanement", BULLET),
          Paragraph("- <b>read_node / read_neighbours / read_outgoing</b> -- inspecte le noeud courant", BULLET),
          Paragraph("- <b>get_call_chain / get_callers</b> -- trace les flux d'execution des methodes", BULLET),
          Paragraph("- <b>read_source_code</b> -- extrait le code Java autour de la declaration", BULLET),
          Paragraph("- <b>find_path</b> -- chemin le plus court entre deux noeuds", BULLET),
          Paragraph("- <b>add_note</b> -- sauvegarde une observation dans la memoire de l'agent", BULLET),
          Spacer(1,0.2*cm)]

    # Etape 4
    s += [badge(4, "MLflow -- Observabilite Complete des Traces"), Spacer(1,0.2*cm),
          Paragraph(
            "Chaque execution est enregistree dans MLflow (experience <i>graph-agent</i>, "
            "<code>http://localhost:5000</code>). Avec <code>log_traces=True</code>, "
            "l'autolog DSPy capture l'arbre hierarchique complet :", BODY),
          Paragraph("- <b>ReAct.forward</b> -- span racine de la boucle de raisonnement", BULLET),
          Paragraph("- <b>Predict.forward -> ChatAdapter -> LM.__call__</b> -- chaque etape LLM", BULLET),
          Paragraph("- <b>Tool.tool_xxx</b> -- chaque appel d'outil avec ses entrees et sorties", BULLET),
          Spacer(1,0.15*cm)]
    s += img_block("mlflow_trace_list.png", 6.5,
                   "Fig. 2a -- Vue liste MLflow : appels d'outils entremeles avec les etapes de raisonnement LLM")
    s += [Spacer(1,0.2*cm)]
    s += img_block("mlflow_trace_graph.png", 15.5,
                   "Fig. 2b -- Vue graphe MLflow : ReAct -> Outils + ChainOfThought -> LM.__call__")
    s += [Spacer(1,0.25*cm)]

    # Etape 5
    s += [badge(5, "Streamlit -- Visualisation Temps Reel"), Spacer(1,0.2*cm),
          Paragraph(
            "<b>graph_agent/visualizer.py</b> lit <code>nav_state.json</code> toutes les secondes "
            "et re-affiche le graphe complet avec un code couleur :", BODY),
          Paragraph("- <b>Hexagone blanc</b> -- noeud courant (l'agent est ici)", BULLET),
          Paragraph("- <b>Point violet</b> -- noeud deja visite", BULLET),
          Paragraph("- <b>Arete teal</b> -- chemin parcouru", BULLET),
          Paragraph("- <b>Point gris</b> -- noeud non visite", BULLET),
          Spacer(1,0.15*cm)]
    s += img_block("streamlit_overview.png", 15.5,
                   "Fig. 3a -- Vue d'ensemble Streamlit : 150 noeuds, 205 aretes -- "
                   "l'agent est sur Visit, 1 deplacement effectue (Visit -> constructeur Visit())")
    s += [Spacer(1,0.2*cm)]
    s += img_block("streamlit_zoom.png", 11,
                   "Fig. 3b -- Zoom : classe Visit (violet = visitee) reliee au constructeur .Visit() "
                   "(hexagone blanc = position courante) via une arete METHOD")
    s += [Spacer(1,0.3*cm)]

    # Exemple de question
    s += [Paragraph("Exemple de Question Posee a l'Agent", H1),
          Paragraph(
            "<i>\"Quel est le flux de donnees complet lorsqu'un utilisateur soumet une nouvelle visite "
            "pour un animal ? Retracez le chemin d'execution complet depuis la couche HTTP jusqu'a la "
            "base de donnees, identifiez toutes les classes et methodes impliquees, toute logique de "
            "validation, et determinez s'il existe un risque de sauvegarder une visite sans "
            "proprietaire valide.\"</i>", BODY),
          Spacer(1,0.15*cm),
          Paragraph(
            "L'agent demarre automatiquement sur <code>owner_visit_visit</code> (mot-cle 'visit' "
            "detecte dans la question), puis explore : <b>classe Visit</b> -> "
            "<b>constructeur Visit()</b> -> voisins -> <b>VisitController</b> -> "
            "<b>processNewVisitForm()</b> -> <b>OwnerRepository</b>, en enregistrant une note a "
            "chaque etape avant de formuler sa reponse finale.", BODY),
          Spacer(1,0.3*cm),
          hr(),
          Paragraph(
            "Code source : C:\\Projet\\graph_agent\\  |  Neo4j : localhost:7687  |  "
            "MLflow : localhost:5000  |  Streamlit : localhost:8501", CAPTION)]

    doc.build(s)
    print(f"PDF genere -> {OUT}")

if __name__ == "__main__":
    build()