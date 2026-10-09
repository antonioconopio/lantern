"""Builds the multi-page PDF in the eval corpus.

The handbook is a PDF on purpose: it's the only way to test page-level
citations. Its text lives here so it can be edited and rebuilt:

    python -m evals.build_corpus

All content in the corpus is fictional, so the model can't answer from
what it already knows; correct answers must come from retrieval.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

CORPUS_DIR = Path(__file__).parent / "corpus"

HANDBOOK_PAGES = [
    (
        "Northwind Cycling Club Member Handbook",
        [
            "Welcome to Northwind Cycling Club. The club was founded in 2011 in "
            "Kestrel Bay and runs group rides from April through October.",
            "Membership tiers. A Standard membership costs $85 per year. A Student "
            "membership costs $40 per year and requires a valid student ID. A "
            "Family membership costs $150 per year and covers up to four people "
            "living in the same household.",
            "Renewals. Memberships must be renewed by March 31 each year. Members "
            "who renew after March 31 pay a $15 reinstatement fee in addition to "
            "their membership price.",
        ],
    ),
    (
        "Group Ride Rules",
        [
            "Pace groups. Every ride is split into three pace groups. The Blue group "
            "is casual, averaging 18 to 22 km/h. The Green group is moderate, "
            "averaging 23 to 27 km/h. The Black group is fast, averaging 28 km/h or more.",
            "Drop policy. Blue and Green are no-drop rides: the group waits for "
            "anyone who falls behind. Black is a drop ride, and riders who cannot "
            "hold the pace are expected to know their own way home.",
            "Equipment. Helmets are mandatory on every ride; riders without a helmet "
            "will not be allowed to start. A working rear light is required for any "
            "ride that starts before 7:00 a.m.",
            "Each group has a ride leader at the front and a designated sweep rider "
            "who stays at the back of the group.",
        ],
    ),
    (
        "Safety and Incidents",
        [
            "Incident reporting. Any crash, injury or near miss must be reported to "
            "the club safety officer within 48 hours using the online incident form.",
            "First aid. Every ride leader carries a first-aid kit. At least one "
            "rider in each group should hold a current first-aid certificate.",
            "Weather. Rides are cancelled if the temperature is below -5 degrees "
            "Celsius at the start time or if lightning is forecast. Cancellations "
            "are posted in the club group chat by 6:00 a.m. on the day of the ride.",
        ],
    ),
    (
        "Events and Volunteering",
        [
            "The Lakeshore Century. The club's signature event is a 160 km ride "
            "held on the second Saturday of September each year.",
            "Volunteering. Every member must volunteer at least 4 hours per season, "
            "for example as a ride marshal or at a rest stop. Members who meet this "
            "requirement receive a 20% discount at the partner shop, Spoke & Gear.",
            "Orientation. New members are invited to an orientation ride held on "
            "the first Sunday of every month during the riding season.",
        ],
    ),
]


def build_handbook(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(HANDBOOK_PAGES):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(heading, styles["Heading1"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


if __name__ == "__main__":
    CORPUS_DIR.mkdir(exist_ok=True)
    out = CORPUS_DIR / "northwind_cycling_handbook.pdf"
    build_handbook(out)
    print(f"Wrote {out}")