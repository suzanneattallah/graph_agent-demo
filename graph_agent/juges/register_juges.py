"""
Enregistre les 4 prompts des juges GraphAgent dans le MLflow Prompt Registry.

Ces prompts évaluent la qualité d'exploration du graphe de code Neo4j (projet mall).
Chaque juge produit un VERDICT + un SCORE numérique (0.0 / 0.5 / 1.0) pour les
courbes MLflow, basé sur des métriques OBJECTIVES extraites de la trace.

Juges :
  - JugeExploration     : profondeur et couverture des layers (metrics: n_visited, n_layers)
  - JugePrecisionTechnique : cohérence trace vs. réponse (metrics: citations_valides, inventions)
  - JugeRaisonnement    : flow architectural et usage des notes (metrics: path_coherence)
  - JugeAmeliorations   : améliorations concrètes avec impact mesurable

Métriques loggées dans MLflow :
  score_exploration, score_precision, score_raisonnement, score_ameliorations
  n_visited, n_layers, pct_satisfaisant

Usage : python -m graph_agent.juges.register_juges
"""

import mlflow

MLFLOW_URL = "http://localhost:5000"
JUDGE_PREFIX = "graph-agent"

# ── Présomption d'innocence (anti-biais "il faut toujours trouver un défaut") ─
# Inspiré du projet Juge_Agent_recherche, adapté à l'exploration de graphe de code.
PRESOMPTION_INNOCENCE = """\
============================================================
PRÉSOMPTION D'INNOCENCE — RÈGLE ABSOLUE À APPLIQUER EN PREMIER
============================================================
Le verdict par défaut est SATISFAISANT. Si l'agent a produit un résultat
correct et globalement utile, le verdict est SATISFAISANT même s'il reste
des améliorations possibles. Une suggestion mineure ne justifie PAS "À AMÉLIORER".

CALIBRATION STRICTE DES 3 NIVEAUX :
- SATISFAISANT (score=1.0) : l'agent a fait son travail. Imperfections mineures
  (un nœud de moins qu'idéal, une note absente, formulation perfectible) → SATISFAISANT.
- À AMÉLIORER  (score=0.5) : il existe un problème CONCRET et NON-TRIVIAL qui DÉGRADE
  la valeur de la réponse (exploration trop courte, layer architectural complètement
  absent, raisonnement cassé, citations hors-trace).
- INSUFFISANT  (score=0.0) : l'agent n'a pas exploré (≤1 nœud), a répondu de mémoire
  sans utiliser les outils, ou la réponse est hors-sujet / fausse.

RÈGLE D'OR : en cas d'hésitation entre SATISFAISANT et À AMÉLIORER → SATISFAISANT.
Le biais naturel des juges LLM est de toujours trouver un défaut : résiste-lui.
============================================================
"""

# ── Grille des layers architecturaux du projet mall ──────────────────────────
# Utilisée par JugeExploration et JugeRaisonnement pour mesurer la couverture.
LAYERS_MALL = """\
Layers architecturaux reconnaissables dans les node IDs du projet mall :
  - CONTROLLER  : préfixe "controller_" (ex: controller_omsportalordercontroller_...)
  - SERVICE     : préfixe "service_" ou "impl_" (ex: impl_omsportalorderserviceimpl_...)
  - MAPPER/REPO : préfixe "mapper_" (ex: mapper_omsordermapper_...)
  - MODEL/ENTITY: préfixe "model_" ou "bo_" ou "dto_" (ex: model_omsorder_...)
  - CONFIG/SEC  : préfixe "config_" ou "security_" ou "component_"
  - DEMO/ENTRY  : préfixe "demo_" (point d'entrée Spring Boot)
"""

FALLBACK_PROMPTS = {

    # ──────────────────────────────────────────────────────────────────────────
    "JugeExploration": f"""Tu es un expert en évaluation d'agents LLM naviguant des graphes de code source.
Tu évalues la PROFONDEUR et la COUVERTURE de l'exploration du graphe Neo4j (projet mall).

{PRESOMPTION_INNOCENCE}

Question posée : {{{{ inputs }}}}
Réponse de l'agent : {{{{ outputs }}}}
Trace d'exploration (nœuds visités, outils appelés, notes) : {{{{ trace }}}}

{LAYERS_MALL}

━━━ ÉTAPE 1 — EXTRACTION DES MÉTRIQUES OBSERVABLES (faits uniquement) ━━━
Lis la trace et note EXACTEMENT :
- N_VISITED  : nombre de nœuds DISTINCTS visités (compte les IDs uniques dans "visited")
- N_LAYERS   : nombre de layers différents couverts parmi Controller/Service/Mapper/Model/Config
  (identifie le préfixe de chaque nœud visité)
- OUTILS_DIV : liste des outils distincts utilisés (move_to, search_node, read_neighbours, etc.)
- NOTES      : nombre de notes enregistrées via add_note (0 si aucune)
- SOURCE_READ: OUI si read_source_code a été appelé au moins une fois, NON sinon

━━━ ÉTAPE 2 — APPLICATION DE LA GRILLE (mécanique, basée uniquement sur les métriques) ━━━
Applique cette grille sans interprétation :
  N_VISITED ≥ 6 ET N_LAYERS ≥ 3  → orientation SATISFAISANT fort
  N_VISITED ≥ 4 ET N_LAYERS ≥ 2  → orientation SATISFAISANT
  N_VISITED 3-4 ET N_LAYERS = 1   → orientation À AMÉLIORER
  N_VISITED ≤ 2 OU N_LAYERS = 0   → orientation INSUFFISANT

━━━ ÉTAPE 3 — PRÉSOMPTION D'INNOCENCE ━━━
Si le résultat est borderline (ex: 4 nœuds, 2 layers), applique SATISFAISANT.
Un agent qui explore, même imparfaitement, mérite SATISFAISANT.

MÉTRIQUES MESURÉES :
N_VISITED=[valeur] | N_LAYERS=[valeur] | NOTES=[valeur] | SOURCE_READ=[OUI/NON]

ANALYSE :
(2-3 phrases factuelles : quels nœuds, quels layers, quelle progression dans le graphe)

VERDICT : [SATISFAISANT / À AMÉLIORER / INSUFFISANT]
SCORE : [1.0 / 0.5 / 0.0]

RECOMMANDATION : (action concrète UNIQUEMENT si N_VISITED ≤ 3 ou N_LAYERS ≤ 1,
sinon exactement : "Exploration suffisante pour la question posée")""",

    # ──────────────────────────────────────────────────────────────────────────
    "JugePrecisionTechnique": f"""Tu es un expert en évaluation d'agents LLM naviguant un graphe de code Java (projet mall).
Tu évalues la PRÉCISION TECHNIQUE : la réponse est-elle ancrée dans ce que l'agent a RÉELLEMENT vu ?

{PRESOMPTION_INNOCENCE}

Question posée : {{{{ inputs }}}}
Réponse de l'agent : {{{{ outputs }}}}
Trace d'exploration (nœuds visités, notes, appels d'outils) : {{{{ trace }}}}

━━━ ÉTAPE 1 — EXTRACTION DES FAITS VÉRIFIABLES ━━━
Lis la trace et la réponse, puis note :
- NODES_VISITED_LIST  : liste des node IDs effectivement visités (depuis la trace)
- NODES_CITED_IN_ANS  : node IDs ou noms de classes/méthodes cités dans la RÉPONSE
- CITATIONS_VALIDES   : citations de la réponse qui correspondent à un nœud visité
- CITATIONS_INVENTEES : citations de la réponse qui n'apparaissent PAS dans la trace
  (attention : ne confonds pas "nœud non visité" avec "erreur" — l'agent peut inférer
   des noms logiques depuis des nœuds voisins vus)
- CONVENTIONS_JAVA    : les noms respectent-ils CamelCase Java ? (OUI/NON/PARTIELLEMENT)

━━━ ÉTAPE 2 — GRILLE D'ÉVALUATION ━━━
  0 citation inventée → orientation SATISFAISANT
  1-2 citations douteuses mais logiquement déductibles → orientation SATISFAISANT
  ≥3 citations clairement inventées OU classe entière inventée → orientation À AMÉLIORER
  Réponse sans aucune référence concrète au graphe → orientation INSUFFISANT

━━━ ÉTAPE 3 — PRÉSOMPTION D'INNOCENCE ━━━
Un nom de méthode légèrement différent mais plausible dans le contexte Java → SATISFAISANT.
Seules les inventions manifestes (classes inexistantes, méthodes impossibles) → À AMÉLIORER.

MÉTRIQUES MESURÉES :
CITATIONS_VALIDES=[n] | CITATIONS_INVENTEES=[n] | CONVENTIONS_JAVA=[OUI/NON/PARTIELLEMENT]

ANALYSE :
(2-3 phrases : quelles citations sont ancrées dans la trace, y a-t-il des inventions ?)

VERDICT : [SATISFAISANT / À AMÉLIORER / INSUFFISANT]
SCORE : [1.0 / 0.5 / 0.0]

RECOMMANDATION : (concrète UNIQUEMENT si ≥3 inventions avérées,
sinon exactement : "Précision technique cohérente avec la trace")""",

    # ──────────────────────────────────────────────────────────────────────────
    "JugeRaisonnement": f"""Tu es un expert en évaluation d'agents LLM naviguant des graphes de code source (projet mall).
Tu évalues la COHÉRENCE LOGIQUE : le chemin de navigation suit-il un raisonnement architectural ?

{PRESOMPTION_INNOCENCE}

Question posée : {{{{ inputs }}}}
Réponse de l'agent : {{{{ outputs }}}}
Trace d'exploration (chemin, notes, appels d'outils) : {{{{ trace }}}}

{LAYERS_MALL}

━━━ ÉTAPE 1 — RECONSTRUCTION DU CHEMIN ━━━
Depuis la trace, reconstitue la séquence de nœuds visités dans l'ordre chronologique.
Identifie le layer de chaque nœud (Controller / Service / Mapper / Model / etc.).

━━━ ÉTAPE 2 — ÉVALUATION DU FLOW ARCHITECTURAL ━━━
Pour chaque type de question, le flow attendu est différent. Évalue si le chemin est
COHÉRENT avec ce qui était demandé :

  Question "trace le flux d'une requête HTTP" → flow attendu : Controller→Service→Mapper
  Question "comment fonctionne X" → l'agent doit visiter X et ses voisins directs
  Question "quelle est la hiérarchie de classes" → l'agent doit suivre les arêtes EXTENDS/IMPLEMENTS

  Un chemin est INCOHÉRENT seulement si :
  - L'agent explore un layer totalement non pertinent pour la question (MAJEUR)
  - L'agent répond à une question différente de celle posée (MAJEUR)
  - La réponse contient une chaîne d'appels fictive non vue dans la trace (MAJEUR)

  Un chemin est ACCEPTABLE si :
  - L'ordre n'est pas parfait mais couvre les éléments pertinents
  - L'agent fait quelques détours mais revient au sujet
  - Des notes sont absentes mais la réponse reste fondée sur les outils

━━━ ÉTAPE 3 — PRÉSOMPTION D'INNOCENCE ━━━
Un chemin sous-optimal mais cohérent → SATISFAISANT.
L'absence de notes seule ne suffit pas pour À AMÉLIORER.

MÉTRIQUES MESURÉES :
FLOW_TYPE=[question type: flux/composant/hiérarchie/cross] | PATH_COHERENT=[OUI/PARTIEL/NON]
REPONSE_CENTREE_QUESTION=[OUI/NON] | NOTES_UTILISEES=[OUI/NON]

ANALYSE :
(2-3 phrases factuelles sur le chemin suivi et son adéquation à la question)

VERDICT : [SATISFAISANT / À AMÉLIORER / INSUFFISANT]
SCORE : [1.0 / 0.5 / 0.0]

RECOMMANDATION : (concrète UNIQUEMENT si PATH_COHERENT=NON ou REPONSE_CENTREE_QUESTION=NON,
sinon exactement : "Raisonnement cohérent avec l'exploration effectuée")""",

    # ──────────────────────────────────────────────────────────────────────────
    "JugeAmeliorations": f"""Tu es un expert en optimisation d'agents LLM d'exploration de graphes de code (projet mall).
Tu identifies les AMÉLIORATIONS À FORT IMPACT sur la qualité d'exploration.

{PRESOMPTION_INNOCENCE}

Question posée : {{{{ inputs }}}}
Réponse de l'agent : {{{{ outputs }}}}
Trace d'exploration (nœuds visités, chemin, notes, outils) : {{{{ trace }}}}

━━━ ÉTAPE 1 — INVENTAIRE DES POINTS FORTS (obligatoire avant critique) ━━━
Liste 1 à 3 choses que l'agent a bien faites (exploration, outils utilisés, réponse structurée).
Si l'agent a correctement exploré ≥5 nœuds et ≥2 layers → c'est déjà un succès.

━━━ ÉTAPE 2 — IDENTIFICATION DES LACUNES MAJEURES ━━━
Une lacune est MAJEURE uniquement si elle répond OUI à toutes ces questions :
  (a) Elle est OBSERVABLE dans la trace (pas une supposition)
  (b) Elle DÉGRADE CONCRÈTEMENT la valeur de la réponse pour l'utilisateur
  (c) Elle aurait pu être résolue par un appel d'outil différent

  Exemples de lacunes MAJEURES :
  ✗ L'agent a posé la question sur le flux de commande mais n'a pas visité le Mapper de persistance
  ✗ L'agent a cherché une classe mais n'a pas utilisé search_node et a raté la cible
  ✗ L'agent n'a lu aucun code source (read_source_code = 0) pour une question d'implémentation

  Exemples de lacunes NON-majeures (cosmétique) :
  ✓ L'agent aurait pu visiter 1 nœud supplémentaire
  ✓ Les notes ne sont pas structurées parfaitement
  ✓ La réponse pourrait être mieux formatée

━━━ ÉTAPE 3 — PRÉSOMPTION D'INNOCENCE ━━━
Si les lacunes ne sont que cosmétiques → SATISFAISANT.
SEULES les lacunes MAJEURES (impact réel, observable, actionnable) → À AMÉLIORER.

POINTS FORTS OBSERVÉS :
(liste les 1-3 points positifs concrets)

LACUNES MAJEURES DÉTECTÉES :
(liste uniquement les lacunes observables et à fort impact, ou "Aucune lacune majeure détectée")

VERDICT : [SATISFAISANT / À AMÉLIORER / INSUFFISANT]
SCORE : [1.0 / 0.5 / 0.0]

RECOMMANDATION :
Si lacunes majeures : liste-les par priorité décroissante d'impact (max 3) avec l'outil/action corrective.
Sinon exactement : "Aucune amélioration majeure nécessaire" """,
}


def register_all_judges():
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlf_client = mlflow.MlflowClient()

    print(f"Enregistrement des juges dans MLflow Prompt Registry ({MLFLOW_URL})...\n")

    for judge_name, template in FALLBACK_PROMPTS.items():
        registry_name = f"{JUDGE_PREFIX}-{judge_name}"
        try:
            # Vérifie si déjà enregistré
            versions = mlf_client.search_prompt_versions(name=registry_name)
            if versions:
                latest = max(int(v.version) for v in versions)
                print(f"  ✓ {registry_name} : déjà dans le Registry (v{latest}) — ignoré")
                continue
        except Exception:
            pass

        try:
            pv = mlflow.genai.register_prompt(
                name=registry_name,
                template=template,
                commit_message="Prompt initial du juge GraphAgent",
                tags={"judge_type": judge_name, "iteration": "0", "status": "initial"},
            )
            print(f"  ✓ {registry_name} v{pv.version} enregistré")
        except Exception as e:
            print(f"  ✗ {registry_name} : échec ({type(e).__name__}: {e})")

    print("\nEnregistrement terminé.")
    print("Lance ensuite : python -m graph_agent.juges.generer_traces")


if __name__ == "__main__":
    register_all_judges()
