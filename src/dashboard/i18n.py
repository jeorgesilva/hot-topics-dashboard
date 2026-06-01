"""German UI string constants for the Hot Topics Dashboard."""
from __future__ import annotations

# ── app-level ──────────────────────────────────────────────────────────────────
PAGE_TITLE = "Hot Topics Dashboard"
APP_TITLE = "🔍 Hot Topics"
APP_CAPTION = "Desinformations-Risiko-Dashboard"

# ── home view ──────────────────────────────────────────────────────────────────
METRIC_TOPICS_ANALYSED = "Analysierte Themen"
METRIC_TOPICS_ANALYSED_HELP = (
    "Gesamtzahl der identifizierten Themencluster. "
    "Jeder Cluster gruppiert Artikel zum selben Thema."
)
METRIC_HIGH_RISK = "Hochrisiko-Themen"
METRIC_AVG_TRUST = "Ø Quellen-Vertrauen"
METRIC_AVG_TRUST_HELP = (
    "Mittlerer Vertrauenswert (0–100) über alle bewerteten Artikel, basierend auf MBFC. "
    "≥ 60 = glaubwürdig · 40–59 = neutral · < 40 = unzuverlässig."
)
METRIC_AVG_RISK = "Ø Gesamtrisiko"
METRIC_AVG_RISK_HELP = (
    "Gewichteter Durchschnitt von 7 NLP- und Abdeckungssignalen. "
    "25 % Quellen-Misstrauen + 20 % Sentiment + 20 % Geringe Abdeckung + "
    "15 % Framing-Divergenz + 10 % Sensationalismus + 5 % Attributions-Vagheit + "
    "5 % Fakten-Inkonsistenz. Höher = mehr Desinformationsrisiko."
)

SECTION_TOPIC_RANKING = "Themen-Ranking"
LABEL_ARTICLES = "Artikel"
LABEL_RISK = "Risiko"
LABEL_RELIABILITY = "Zuverlässigkeit"
LABEL_UNSCORED = "nicht bewertet"
LABEL_DIV_TOOLTIP = (
    "Narrative Divergenz: wie stark sich Reddit-Framing von verifizierten Quellen unterscheidet"
)

CAPTION_PIPELINE_MISSING = (
    "Keine bewerteten Themen gefunden unter `{db_path}`.\n\n"
    "Pipeline zuerst ausführen:\n"
    "```\npython src/scrapers/run_all.py\n"
    "python src/scoring/compute_scores.py\n```"
)

CAPTION_ONLY_SCORED = (
    "Nur vollständig bewertete Themen (NLP-Pipeline abgeschlossen) sind hier aufgeführt."
)

# ── charts (home) ──────────────────────────────────────────────────────────────
CHART_SENTIMENT_VS_SENS = "Sentiment vs. Sensationalismus"
CHART_COMPOSITE_RISK = "Gesamtrisiko nach Thema"
CHART_RISK_THRESHOLD_LABEL = "Schwellenwert"
AXIS_SENTIMENT = "Sentiment-Extremität"
AXIS_SENSATIONALISM = "Sensationalismus"
AXIS_RISK = "Risiko"
AXIS_COMPOSITE_RISK = "Gesamtrisiko (0–100 %)"

# ── topic detail ───────────────────────────────────────────────────────────────
BTN_BACK = "← Zurück"
TOPIC_NOT_FOUND = "Thema {topic_id} nicht gefunden."
TOPIC_NOT_SCORED = (
    "Dieses Thema wurde noch nicht bewertet. "
    "Führen Sie `python src/scoring/compute_scores.py` aus."
)
CAPTION_COMPOSITE_RISK = "Gesamtrisiko: **{risk:.1f} %**"
CAPTION_ARTICLES = "{n} Artikel"
CAPTION_SCORED_AT = "bewertet {ts}"

SECTION_SIGNAL_BREAKDOWN = "Signal-Aufschlüsselung"
SIGNAL_BREAKDOWN_CAPTION = "Fahren Sie mit der Maus über eine Karte für eine vollständige Erklärung."

SECTION_SOCIAL_TRACK = "Social-Media-Spur (Reddit)"
SOCIAL_NO_DATA = (
    "Keine Reddit-Artikel mit diesem Thema verknüpft — Social-Risiko-Spur nicht verfügbar. "
    "Führen Sie die Pipeline mit aktiviertem Reddit aus, um diesen Bereich zu befüllen."
)

METRIC_VERIFIED_RISK = "Verifiziertes Risiko"
METRIC_VERIFIED_RISK_HELP = "Gesamtrisiko aus NewsAPI/RSS-Artikeln (journalistische Quellen)."
METRIC_SOCIAL_RISK = "Soziales Risiko"
METRIC_SOCIAL_RISK_HELP = "Gesamtrisiko aus Reddit-Beiträgen zu diesem Thema."
METRIC_NARRATIVE_DIV = "Narrative Divergenz"
METRIC_NARRATIVE_DIV_HELP = (
    "Absoluter Abstand zwischen verifiziertem und sozialem Risiko. "
    "Hohe Divergenz bedeutet, dass Reddit-Diskussionen das Thema sehr anders rahmen "
    "als journalistische Quellen — ein potenzielles Desinformationssignal."
)

DIVERGENCE_HIGH = "hohe Divergenz — Hinweis auf koordinierte soziale Verstärkung."
DIVERGENCE_MED = "moderate Divergenz — soziales Framing weicht von der Presseberichterstattung ab."
DIVERGENCE_LOW = "geringe Divergenz — soziale und Presseberichterstattung weitgehend übereinstimmend."
DIVERGENCE_PREFIX = "Narrative Divergenz"
DIVERGENCE_VS = "Verifiziertes Risiko {v:.1f} % vs. Soziales Risiko {s:.1f} %"

CAPTION_SOCIAL_GRADE = "Soziale Note:"
CAPTION_SOCIAL_BASED_ON = "Basierend auf Reddit-Beiträgen zu diesem Thema."

SECTION_RISK_RADAR = "Risiko-Radar"
RADAR_CHART_TITLE = "Risiko-Radar"
RADAR_CAPTION = (
    "Jede Achse zeigt ein normiertes Risikosignal (0–100 %). "
    "Eine größere Fläche = höheres Desinformationsrisiko. "
    "Fahren Sie mit der Maus über einen Punkt, um den genauen Wert zu sehen."
)
RADAR_CATEGORIES: list[str] = [
    "Quellen-Misstrauen",
    "Sentiment",
    "Geringe Abdeckung",
    "Framing",
    "Sensationalismus",
]

SECTION_DOMAIN_TRUST = "Domänen-Vertrauenswerte"
DOMAIN_TRUST_CAPTION = (
    "Vertrauenswerte aus Media Bias/Fact Check (MBFC). "
    "🟢 ≥ 60 = glaubwürdig · 🟠 40–59 = neutral · 🔴 < 40 = unzuverlässig. "
    "Hohe Streuung zwischen Grün und Rot ist ein Desinformationssignal."
)
DOMAIN_TRUST_NO_DATA = "Keine Domänendaten verfügbar."

SECTION_ARTICLES = "Artikel"
ARTICLES_NONE = "Keine Artikel für dieses Thema gefunden."

# ── signal names ───────────────────────────────────────────────────────────────
SIGNAL_NAMES: dict[str, str] = {
    "Source Distrust":     "🏛️ Quellen-Misstrauen",
    "Sentiment Extremity": "😤 Sentiment-Extremität",
    "Low Coverage":        "📡 Geringe Abdeckung",
    "Framing Divergence":  "🔀 Framing-Divergenz",
    "Sensationalism":      "📢 Sensationalismus",
    "Attribution Vagueness": "⚠️ Attributions-Vagheit",
    "Fact Inconsistency":  "📋 Fakten-Inkonsistenz",
}

SIGNAL_DETAIL_LABELS: dict[str, str] = {
    "Source Distrust":     "Ø Vertrauen {val:.1f}",
    "Sentiment Extremity": "Signal {val:.1f} %",
    "Low Coverage":        "{val:.1f} % glaubwürdige Domains",
    "Framing Divergence":  "Signal {val:.1f} %",
    "Sensationalism":      "Signal {val:.1f} %",
}

# ── signal tooltips (German) ───────────────────────────────────────────────────
SIGNAL_TOOLTIPS: dict[str, str] = {
    "Source Distrust": (
        "Misst, wie viel der Themenberichterstattung aus wenig vertrauenswürdigen Quellen stammt. "
        "Gewicht: 25 % des Gesamtrisikos. "
        "Hoch = die meisten Artikel stammen aus unzuverlässigen Medien. "
        "Basiert auf MBFC-Vertrauenswerten (0–100) pro Domain."
    ),
    "Sentiment Extremity": (
        "Durchschnittliche emotionale Intensität der Artikel — wie stark das Sentiment vom Neutral abweicht. "
        "Gewicht: 20 % des Gesamtrisikos. "
        "Hoch = Artikel verwenden stark polarisierte, emotional aufgeladene Sprache. "
        "Berechnet mit einem deutschen BERT-Sentimentmodell (oliverguhr/german-sentiment-bert)."
    ),
    "Low Coverage": (
        "Anteil der Themenberichterstattung aus unglaubwürdigen Domains. "
        "Gewicht: 20 % des Gesamtrisikos. "
        "Hoch = Geschichte wird nur von wenig vertrauenswürdigen Quellen aufgegriffen — "
        "ein klassisches Desinformationsmuster."
    ),
    "Framing Divergence": (
        "Wie unterschiedlich vertrauensstarke vs. schwache Quellen das gleiche Thema einrahmen. "
        "Gewicht: 15 % des Gesamtrisikos. "
        "Gemessen als Kosinus-Distanz zwischen deutschen RoBERTa-Embeddings der zwei Quell-Tiers. "
        "Hoch = glaubwürdige und unzuverlässige Quellen erzählen sehr verschiedene Geschichten."
    ),
    "Sensationalism": (
        "Dichte von GROSSBUCHSTABEN, Ausrufezeichen, reißerischen Begriffen (z. B. 'Schock', 'Skandal') "
        "und Clickbait-Mustern über alle Artikel. "
        "Gewicht: 10 % des Gesamtrisikos. "
        "Hoch = starke sensationalistische Rhetorik."
    ),
}

# ── breakdown chart ────────────────────────────────────────────────────────────
BREAKDOWN_CHART_TITLE = "Risikobeitrag je Signal"
BREAKDOWN_CHART_CAPTION = (
    "Der gestapelte Balken zeigt den gewichteten Beitrag jedes Signals zum Gesamtrisiko. "
    "Längere Segmente = größerer Einfluss auf das Risiko."
)
BREAKDOWN_SIGNAL_COLOURS: list[str] = [
    "#3498db",  # Source Distrust
    "#e74c3c",  # Sentiment Extremity
    "#9b59b6",  # Low Coverage
    "#e67e22",  # Framing Divergence
    "#f1c40f",  # Sensationalism
    "#1abc9c",  # Attribution Vagueness
    "#e91e63",  # Fact Inconsistency
]

# ── article view ───────────────────────────────────────────────────────────────
SECTION_SIGNAL_ANALYSIS = "Signalanalyse"
ARTICLE_ANALYSIS_LABEL = "Artikel-Analyse"
GAUGE_SENSATIONALISM = "Sensationalismus"
GAUGE_ATTRIBUTION = "Attributions-Vagheit"
GAUGE_CLICKBAIT = "Clickbait-Muster"
GAUGE_CAPS = "Großbuchstaben-Dichte"

ARTICLE_FULL_TEXT = "Vollständiger Artikeltext"
ARTICLE_SUMMARY = "Zusammenfassung"
ARTICLE_DESCRIPTION = "Beschreibung"
ARTICLE_NO_TEXT = "*Kein Text verfügbar.*"

# ── expander explanations ──────────────────────────────────────────────────────
EXPANDER_HOW_RISK = "ℹ️ Wie wird das Risiko berechnet?"
EXPANDER_HOW_RISK_TEXT = """\
Das **Gesamtrisiko** (0–100 %) ist eine gewichtete Summe aus 7 NLP-Signalen:

| Signal | Gewicht | Beschreibung |
|---|---|---|
| 🏛️ Quellen-Misstrauen | 25 % | Anteil unzuverlässiger Berichterstattung |
| 😤 Sentiment-Extremität | 20 % | Emotionale Intensität der Artikel |
| 📡 Geringe Abdeckung | 20 % | Anteil aus unglaubwürdigen Domains |
| 🔀 Framing-Divergenz | 15 % | Unterschied in der Darstellung je Quell-Tier |
| 📢 Sensationalismus | 10 % | Reißerische Sprache und Clickbait |
| ⚠️ Attributions-Vagheit | 5 % | Vage Quellenangaben ("Experten sagen …") |
| 📋 Fakten-Inkonsistenz | 5 % | Abweichung genannter Entitäten zwischen Quellen |

Ein Risiko ≥ 50 % wird als potenzielles Desinformationssignal markiert.
"""

EXPANDER_SOCIAL_TRACK = "ℹ️ Was ist die Social-Media-Spur?"
EXPANDER_SOCIAL_TRACK_TEXT = """\
Die **Social-Media-Spur** bewertet Reddit-Beiträge getrennt von journalistischen Quellen.

- **Verifiziertes Risiko** — aus NewsAPI/RSS/Google News berechnet
- **Soziales Risiko** — aus Reddit-Beiträgen berechnet
- **Narrative Divergenz** — |verifiziertes − soziales Risiko|

Hohe Divergenz (≥ 30 %) kann auf koordinierte soziale Verstärkung oder **Narrative Hijacking** hinweisen:
Reddit-Diskussionen rahmen das Thema fundamental anders als die Presse.
"""

EXPANDER_DOMAIN_TRUST = "ℹ️ Woher stammen die Vertrauenswerte?"
EXPANDER_DOMAIN_TRUST_TEXT = """\
Vertrauenswerte werden in dieser Reihenfolge aufgelöst:

1. **Media Bias/Fact Check (MBFC)** — manuell kuratierte Datenbank mit 100+ deutschen und internationalen Nachrichtenmedien
2. **TLD-Heuristik** — Domains außerhalb der MBFC-Liste erhalten einen Wert basierend auf ihrer Top-Level-Domain
   (.gov = 82 · .edu = 78 · .de/.at/.ch = 52 · .com = 46 · .xyz/.top = 30–32)
3. **Standardwert** — unbekannte Domains erhalten 45 (leichte Abwertung gegenüber neutral)

Schwellen: 🟢 ≥ 60 glaubwürdig · 🟠 40–59 neutral · 🔴 < 40 unzuverlässig
"""

EXPANDER_RADAR = "ℹ️ Was zeigt der Risiko-Radar?"
EXPANDER_RADAR_TEXT = """\
Der Radar-Chart visualisiert 5 Risikosignale gleichzeitig:

- **Quellen-Misstrauen** — wie hoch der Anteil unzuverlässiger Quellen ist
- **Sentiment** — emotionale Intensität der Berichterstattung
- **Geringe Abdeckung** — wie viele Domains nicht glaubwürdig sind
- **Framing** — wie unterschiedlich verschiedene Quellen das Thema rahmen
- **Sensationalismus** — Dichte reißerischer Sprache

Eine große Fläche = hohes Gesamtrisiko. Ein spitzer Ausreißer auf einer einzelnen Achse zeigt, welches Signal das Risiko treibt.
"""
