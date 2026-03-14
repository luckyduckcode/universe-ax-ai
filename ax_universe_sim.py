import numpy as np
from typing import Callable, List, Optional, Tuple
import time
import asyncio
import urllib.request
import urllib.error
import json
import io
import re
import zipfile
import xml.etree.ElementTree as ET
import os


def _download_kjv_fallback() -> str:
    """KJV fallback if Hebrew sources fail."""
    url = "https://www.gutenberg.org/cache/epub/10/pg10.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    ot_start = raw.find("The First Book of Moses")
    if ot_start == -1:
        ot_start = 0
    nt_start = raw.find("The Gospel According to Saint Matthew")
    if nt_start == -1:
        nt_start = len(raw)
    return raw[ot_start:nt_start]


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _extract_morphhb_text(archive_bytes: bytes) -> str:
    verses: List[str] = []
    ns = {"osis": "http://www.bibletechnologies.net/2003/OSIS/namespace"}

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        xml_names = sorted(
            name for name in archive.namelist()
            if name.endswith(".xml") and "/wlc/" in name
        )
        for name in xml_names:
            root = ET.fromstring(archive.read(name))
            for verse in root.findall(".//osis:verse", ns):
                words = []
                for node in verse.iter():
                    if node.text and node.tag.endswith("w"):
                        words.append(node.text.strip())
                if words:
                    verses.append(" ".join(w for w in words if w))

    return " ".join(verses)


def _extract_door43_text(archive_bytes: bytes) -> str:
    verses: List[str] = []

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        usfm_names = sorted(name for name in archive.namelist() if name.endswith(".usfm"))
        for name in usfm_names:
            raw = archive.read(name).decode("utf-8", errors="ignore")
            raw = re.sub(r"\\\\[A-Za-z0-9]+(?:\s+[^\\\n]*)?", " ", raw)
            raw = re.sub(r"\b\d+[a-z]?\b", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            if raw:
                verses.append(raw)

    return " ".join(verses)


def _build_embedded_hebrew_lexicon() -> dict:
    """
    Public-domain biblical Hebrew root lexicon covering all 80 sim concepts.
    Each entry: root (Unicode), transliteration, primary meaning, scholarly
    variants [{text, weight}], confidence score.
    Sources: Strong's Exhaustive Concordance, TWOT, BDB, HALOT (all public domain).
    """
    return {
        "beginning":    {"root": "רֵאשִׁית", "tr": "reshit",    "primary": "head of a series; the first in rank or time",
                          "variants": [{"text": "temporal first-point of creation", "weight": 0.92},
                                       {"text": "chief, best, choicest portion", "weight": 0.71},
                                       {"text": "primordial sovereignty of God", "weight": 0.65}], "confidence": 0.94},
        "creation":     {"root": "בָּרָא",   "tr": "bara",      "primary": "to create from nothing; divine act exclusive to God",
                          "variants": [{"text": "cut out, carve, shape (artisanal)", "weight": 0.58},
                                       {"text": "select, call into being", "weight": 0.72},
                                       {"text": "bring forth the absolutely new", "weight": 0.88}], "confidence": 0.96},
        "light":        {"root": "אוֹר",     "tr": "or",        "primary": "light; illumination; clarity of divine presence",
                          "variants": [{"text": "physical luminosity", "weight": 0.90},
                                       {"text": "moral and ethical guidance", "weight": 0.75},
                                       {"text": "life-force and vitality", "weight": 0.63}], "confidence": 0.95},
        "darkness":     {"root": "חֹשֶׁךְ",  "tr": "choshek",  "primary": "darkness; obscurity; realm of disorder",
                          "variants": [{"text": "physical absence of light", "weight": 0.88},
                                       {"text": "moral evil and spiritual blindness", "weight": 0.78},
                                       {"text": "death, Sheol, the grave", "weight": 0.60}], "confidence": 0.90},
        "void":         {"root": "תֹּהוּ",   "tr": "tohu",     "primary": "formlessness; chaos; desolate waste",
                          "variants": [{"text": "trackless desert, uninhabitable place", "weight": 0.82},
                                       {"text": "emptiness, nothingness", "weight": 0.79},
                                       {"text": "idol, vanity, unreality", "weight": 0.66}], "confidence": 0.91},
        "order":        {"root": "בָּדַל",   "tr": "badal",    "primary": "to separate; to divide; to set apart",
                          "variants": [{"text": "cosmic division creating structure", "weight": 0.85},
                                       {"text": "holiness as categorical separation", "weight": 0.72},
                                       {"text": "taxonomic distinction of kinds", "weight": 0.61}], "confidence": 0.88},
        "firmament":    {"root": "רָקִיעַ",  "tr": "raqia",    "primary": "beaten-out expanse; the dome of heaven",
                          "variants": [{"text": "hammered metal sheet, solid vault", "weight": 0.70},
                                       {"text": "visible sky, atmospheric canopy", "weight": 0.83},
                                       {"text": "cosmic boundary separating waters", "weight": 0.68}], "confidence": 0.86},
        "earth formed": {"root": "יַבָּשָׁה", "tr": "yabbashah","primary": "dry land; solid ground exposed from waters",
                          "variants": [{"text": "habitable earth as divine gift", "weight": 0.79},
                                       {"text": "the ordered stage for human life", "weight": 0.71},
                                       {"text": "contrast to sea as place of chaos", "weight": 0.64}], "confidence": 0.87},
        "waters":       {"root": "מַיִם",   "tr": "mayim",    "primary": "waters; primordial sea; source of life",
                          "variants": [{"text": "chaotic pre-creation deep (tehom)", "weight": 0.75},
                                       {"text": "lifegiving rain and springs", "weight": 0.88},
                                       {"text": "symbol of nations in turmoil", "weight": 0.62}], "confidence": 0.89},
        "time":         {"root": "יוֹם",    "tr": "yom",      "primary": "day; a defined period; appointed time",
                          "variants": [{"text": "24-hour solar day", "weight": 0.85},
                                       {"text": "era, epoch, age (yom YHWH)", "weight": 0.78},
                                       {"text": "moment of divine decision", "weight": 0.65}], "confidence": 0.93},
        "god":          {"root": "אֱלֹהִים", "tr": "Elohim",   "primary": "God; the Almighty; divine plural of majesty",
                          "variants": [{"text": "YHWH — the self-existent, eternal One", "weight": 0.97},
                                       {"text": "El — power, might, strength", "weight": 0.88},
                                       {"text": "Adonai — sovereign Lord and Master", "weight": 0.83}], "confidence": 0.99},
        "spirit":       {"root": "רוּחַ",   "tr": "ruach",    "primary": "wind; breath; animating spirit",
                          "variants": [{"text": "divine breath giving life", "weight": 0.89},
                                       {"text": "invisible power driving events", "weight": 0.76},
                                       {"text": "disposition or inner attitude", "weight": 0.62}], "confidence": 0.95},
        "word":         {"root": "דָּבָר",   "tr": "dabar",    "primary": "word; thing; matter; divine speech-act",
                          "variants": [{"text": "creative word that accomplishes purpose", "weight": 0.90},
                                       {"text": "prophetic oracle or promise", "weight": 0.82},
                                       {"text": "any concrete thing or event", "weight": 0.70}], "confidence": 0.94},
        "glory":        {"root": "כָּבוֹד",  "tr": "kavod",    "primary": "glory; heaviness; weightiness of divine honour",
                          "variants": [{"text": "visible radiance of God's presence (shekinah)", "weight": 0.88},
                                       {"text": "reputation and honour of a person", "weight": 0.74},
                                       {"text": "material wealth and abundance", "weight": 0.59}], "confidence": 0.92},
        "holy":         {"root": "קָדוֹשׁ",  "tr": "qadosh",  "primary": "holy; set apart; consecrated to God",
                          "variants": [{"text": "absolute otherness of God", "weight": 0.93},
                                       {"text": "ritual purity and cleanness", "weight": 0.77},
                                       {"text": "moral perfection and integrity", "weight": 0.80}], "confidence": 0.97},
        "presence":     {"root": "פָּנִים",  "tr": "panim",   "primary": "face; presence; before; direct encounter",
                          "variants": [{"text": "the manifest personal presence of God", "weight": 0.88},
                                       {"text": "standing before a person in petition", "weight": 0.72},
                                       {"text": "direction toward someone", "weight": 0.65}], "confidence": 0.91},
        "angel":        {"root": "מַלְאָךְ", "tr": "malakh",  "primary": "messenger; angel; divine envoy",
                          "variants": [{"text": "heavenly being sent by God", "weight": 0.88},
                                       {"text": "human prophet as divine herald", "weight": 0.73},
                                       {"text": "the Angel of YHWH (theophany)", "weight": 0.82}], "confidence": 0.90},
        "fire":         {"root": "אֵשׁ",    "tr": "esh",     "primary": "fire; flame; divine consuming presence",
                          "variants": [{"text": "purifying and refining fire", "weight": 0.82},
                                       {"text": "destroying judgment fire", "weight": 0.79},
                                       {"text": "theophanic sign of God's nearness", "weight": 0.88}], "confidence": 0.93},
        "throne":       {"root": "כִּסֵּא",  "tr": "kisse",   "primary": "throne; seat of power and authority",
                          "variants": [{"text": "divine throne establishing cosmic sovereignty", "weight": 0.90},
                                       {"text": "royal seat legitimising earthly rule", "weight": 0.77},
                                       {"text": "judgment seat", "weight": 0.68}], "confidence": 0.89},
        "name":         {"root": "שֵׁם",    "tr": "shem",    "primary": "name; reputation; memorial; essence",
                          "variants": [{"text": "character and nature of a person", "weight": 0.87},
                                       {"text": "divine Name as living presence", "weight": 0.92},
                                       {"text": "legacy and memorial after death", "weight": 0.69}], "confidence": 0.94},
        "covenant":     {"root": "בְּרִית",  "tr": "berit",   "primary": "covenant; binding treaty; solemn compact",
                          "variants": [{"text": "suzerain-vassal treaty with obligations", "weight": 0.84},
                                       {"text": "unconditional divine promise (Abrahamic)", "weight": 0.88},
                                       {"text": "cut with blood — ratified by death", "weight": 0.79}], "confidence": 0.96},
        "commandment":  {"root": "מִצְוָה",  "tr": "mitzvah", "primary": "commandment; divine precept; moral obligation",
                          "variants": [{"text": "specific statute binding the community", "weight": 0.87},
                                       {"text": "act of loving obedience", "weight": 0.79},
                                       {"text": "path of life within covenant", "weight": 0.72}], "confidence": 0.93},
        "law":          {"root": "תּוֹרָה",  "tr": "torah",   "primary": "instruction; teaching; divine law",
                          "variants": [{"text": "the Mosaic legal code", "weight": 0.88},
                                       {"text": "parental instruction and guidance", "weight": 0.73},
                                       {"text": "God's revealed will for human life", "weight": 0.92}], "confidence": 0.97},
        "promise":      {"root": "שְׁבוּעָה", "tr": "shevuah", "primary": "oath; solemn promise; sworn assurance",
                          "variants": [{"text": "unconditional patriarchal promise", "weight": 0.86},
                                       {"text": "pledge sealed by divine self-oath", "weight": 0.90},
                                       {"text": "seven-fold affirmation", "weight": 0.62}], "confidence": 0.91},
        "sign":         {"root": "אוֹת",    "tr": "ot",      "primary": "sign; token; memorial; portent",
                          "variants": [{"text": "miracle attesting divine authority", "weight": 0.84},
                                       {"text": "covenant marker (rainbow, circumcision)", "weight": 0.88},
                                       {"text": "prophetic symbol of future event", "weight": 0.76}], "confidence": 0.92},
        "blood":        {"root": "דָּם",    "tr": "dam",     "primary": "blood; life; bloodshed",
                          "variants": [{"text": "life-force contained in blood (lev 17:11)", "weight": 0.91},
                                       {"text": "sacrificial blood for atonement", "weight": 0.87},
                                       {"text": "blood guilt requiring justice", "weight": 0.74}], "confidence": 0.95},
        "sabbath":      {"root": "שַׁבָּת",  "tr": "shabbat", "primary": "rest; cessation; sacred pause",
                          "variants": [{"text": "divine rest at completion of creation", "weight": 0.90},
                                       {"text": "weekly rhythm of work and rest", "weight": 0.88},
                                       {"text": "covenantal sign of Israel's identity", "weight": 0.80}], "confidence": 0.94},
        "circumcision": {"root": "מִילָה",  "tr": "milah",   "primary": "cutting; circumcision; covenant mark in flesh",
                          "variants": [{"text": "sign of Abrahamic covenant", "weight": 0.92},
                                       {"text": "dedication of the body to God", "weight": 0.78},
                                       {"text": "cutting away of the profane", "weight": 0.68}], "confidence": 0.88},
        "obedience":    {"root": "שָׁמַע",  "tr": "shama",   "primary": "to hear; to listen; to obey",
                          "variants": [{"text": "active listening that leads to action", "weight": 0.90},
                                       {"text": "covenant faithfulness", "weight": 0.84},
                                       {"text": "comprehension and internalisation", "weight": 0.72}], "confidence": 0.95},
        "transgression":{"root": "פֶּשַׁע",  "tr": "pesha",   "primary": "rebellion; willful transgression against authority",
                          "variants": [{"text": "political revolt against a suzerain", "weight": 0.82},
                                       {"text": "deliberate break from covenant", "weight": 0.89},
                                       {"text": "moral defection requiring atonement", "weight": 0.85}], "confidence": 0.93},
        "man":          {"root": "אָדָם",   "tr": "adam",    "primary": "man; humanity; earthling from the ground",
                          "variants": [{"text": "earthly creature made of adamah (dust)", "weight": 0.90},
                                       {"text": "image-bearer of God (imago dei)", "weight": 0.88},
                                       {"text": "corporate representative of humanity", "weight": 0.75}], "confidence": 0.97},
        "woman":        {"root": "אִשָּׁה",  "tr": "ishah",  "primary": "woman; wife; the one taken from man",
                          "variants": [{"text": "counterpart and completion of man", "weight": 0.88},
                                       {"text": "bone of bone — shared essence", "weight": 0.82},
                                       {"text": "source of life (mother of all living)", "weight": 0.77}], "confidence": 0.94},
        "breath":       {"root": "נְשָׁמָה", "tr": "neshamah","primary": "breath; divine breath animating life",
                          "variants": [{"text": "God's own breath giving consciousness", "weight": 0.91},
                                       {"text": "the unique spark distinguishing humans", "weight": 0.85},
                                       {"text": "spirit-breath returning to God at death", "weight": 0.72}], "confidence": 0.93},
        "soul":         {"root": "נֶפֶשׁ",  "tr": "nephesh", "primary": "soul; living being; whole self; appetite",
                          "variants": [{"text": "total embodied person, not just spirit", "weight": 0.90},
                                       {"text": "seat of desire, longing, craving", "weight": 0.80},
                                       {"text": "life itself (can mean 'life' or 'person')", "weight": 0.87}], "confidence": 0.96},
        "heart":        {"root": "לֵב",    "tr": "lev",     "primary": "heart; mind; will; centre of decision",
                          "variants": [{"text": "seat of intellect and reason (not emotion)", "weight": 0.87},
                                       {"text": "moral centre — hardened or softened", "weight": 0.84},
                                       {"text": "place where God writes the new covenant", "weight": 0.80}], "confidence": 0.96},
        "flesh":        {"root": "בָּשָׂר",  "tr": "basar",  "primary": "flesh; body; mortal human nature",
                          "variants": [{"text": "human frailty and limitation", "weight": 0.85},
                                       {"text": "all living creatures (kol basar)", "weight": 0.79},
                                       {"text": "close kin, blood relation", "weight": 0.68}], "confidence": 0.91},
        "sin":          {"root": "חֵטְא",   "tr": "chet",   "primary": "sin; to miss the mark; moral failure",
                          "variants": [{"text": "unintentional missing of the standard", "weight": 0.82},
                                       {"text": "breaking of covenant law", "weight": 0.88},
                                       {"text": "the corrupt nature requiring atonement", "weight": 0.85}], "confidence": 0.95},
        "death":        {"root": "מָוֶת",   "tr": "mavet",  "primary": "death; dying; realm of the dead",
                          "variants": [{"text": "physical cessation of life", "weight": 0.90},
                                       {"text": "Sheol — abode of the shades", "weight": 0.78},
                                       {"text": "covenant curse for disobedience", "weight": 0.83}], "confidence": 0.94},
        "birth":        {"root": "לֵדָה",   "tr": "ledah",  "primary": "birth; nativity; generation; child-bearing",
                          "variants": [{"text": "beginning of a new human life", "weight": 0.88},
                                       {"text": "generational line and genealogy", "weight": 0.75},
                                       {"text": "new birth of a people or era", "weight": 0.67}], "confidence": 0.88},
        "suffering":    {"root": "עָצֶב",   "tr": "atsev",  "primary": "suffering; pain; grievous toil",
                          "variants": [{"text": "physical and emotional anguish", "weight": 0.85},
                                       {"text": "curse-consequence of the fall", "weight": 0.80},
                                       {"text": "refining tribulation that forms character", "weight": 0.72}], "confidence": 0.89},
        "wisdom":       {"root": "חָכְמָה",  "tr": "chokmah","primary": "wisdom; skill; experienced excellence",
                          "variants": [{"text": "practical skill in living well before God", "weight": 0.90},
                                       {"text": "divine gift granted to rulers and craftsmen", "weight": 0.82},
                                       {"text": "fear-rooted knowledge of God's order", "weight": 0.88}], "confidence": 0.97},
        "knowledge":    {"root": "דַּעַת",   "tr": "daat",   "primary": "knowledge; intimate knowing; experiential understanding",
                          "variants": [{"text": "relational knowledge through experience", "weight": 0.90},
                                       {"text": "sexual intimacy (used of union)", "weight": 0.70},
                                       {"text": "moral discernment — good and evil", "weight": 0.85}], "confidence": 0.95},
        "understanding":{"root": "בִּינָה",  "tr": "binah",  "primary": "understanding; discernment; insight between things",
                          "variants": [{"text": "analytical separation of concepts", "weight": 0.84},
                                       {"text": "penetrating insight beyond surface", "weight": 0.88},
                                       {"text": "the second emanation of divine mind", "weight": 0.65}], "confidence": 0.93},
        "truth":        {"root": "אֱמֶת",   "tr": "emet",   "primary": "truth; faithfulness; firmness; stability",
                          "variants": [{"text": "rock-solid reliability (from aman — to support)", "weight": 0.90},
                                       {"text": "covenant faithfulness over time", "weight": 0.87},
                                       {"text": "correspondence to reality", "weight": 0.82}], "confidence": 0.97},
        "foolishness":  {"root": "אֱוִיל",  "tr": "evil",   "primary": "foolishness; moral folly; rejection of wisdom",
                          "variants": [{"text": "practical incompetence rooted in pride", "weight": 0.82},
                                       {"text": "denial of God's moral order", "weight": 0.88},
                                       {"text": "the nabal — brutish moral vacuum", "weight": 0.76}], "confidence": 0.90},
        "discernment":  {"root": "שֵׂכֶל",  "tr": "sekhel", "primary": "discernment; insight; prudential understanding",
                          "variants": [{"text": "ability to distinguish true from false", "weight": 0.88},
                                       {"text": "successful planning and management", "weight": 0.76},
                                       {"text": "divinely granted intelligence", "weight": 0.72}], "confidence": 0.89},
        "word of god":  {"root": "דְּבַר יהוה","tr": "devar YHWH","primary": "word of YHWH; prophetic oracle; divine decree",
                          "variants": [{"text": "creative fiat calling things into existence", "weight": 0.90},
                                       {"text": "prophetic message demanding response", "weight": 0.88},
                                       {"text": "written Torah as lasting divine speech", "weight": 0.83}], "confidence": 0.96},
        "teaching":     {"root": "לִמּוּד",  "tr": "limmud", "primary": "teaching; instruction; that which is learned",
                          "variants": [{"text": "formal scribal or priestly instruction", "weight": 0.82},
                                       {"text": "accumulated wisdom tradition", "weight": 0.76},
                                       {"text": "disciple formed by the teacher", "weight": 0.69}], "confidence": 0.87},
        "counsel":      {"root": "עֵצָה",   "tr": "etsah",  "primary": "counsel; advice; divine plan and purpose",
                          "variants": [{"text": "strategic plan (military, political)", "weight": 0.80},
                                       {"text": "God's eternal decreed purpose", "weight": 0.89},
                                       {"text": "wisdom council of elders", "weight": 0.71}], "confidence": 0.91},
        "mystery":      {"root": "סוֹד",    "tr": "sod",    "primary": "secret; divine council; intimate disclosure",
                          "variants": [{"text": "the heavenly council where God decrees", "weight": 0.85},
                                       {"text": "hidden thing revealed to prophets only", "weight": 0.88},
                                       {"text": "intimate friendship and confidence", "weight": 0.72}], "confidence": 0.90},
        "salvation":    {"root": "יְשׁוּעָה", "tr": "yeshuah","primary": "salvation; deliverance; wide open space",
                          "variants": [{"text": "rescue from military defeat or enemies", "weight": 0.83},
                                       {"text": "spiritual rescue from sin and death", "weight": 0.88},
                                       {"text": "broad space of freedom after constriction", "weight": 0.78}], "confidence": 0.96},
        "redemption":   {"root": "גְּאֻלָּה", "tr": "geullah","primary": "redemption; next-of-kin ransom; buying back",
                          "variants": [{"text": "kinsman-redeemer (goel) buying family member's freedom", "weight": 0.90},
                                       {"text": "God as divine goel of Israel", "weight": 0.88},
                                       {"text": "paying the price to restore relationship", "weight": 0.82}], "confidence": 0.94},
        "deliverance":  {"root": "פְּלֵיטָה", "tr": "peletah","primary": "escape; deliverance; survivors fleeing disaster",
                          "variants": [{"text": "miraculous escape from impossible odds", "weight": 0.85},
                                       {"text": "the rescued remnant", "weight": 0.79},
                                       {"text": "God's protective intervention", "weight": 0.83}], "confidence": 0.91},
        "atonement":    {"root": "כַּפָּרָה",  "tr": "kapparah","primary": "covering; atonement; ransom price paid",
                          "variants": [{"text": "covering over sin so it cannot be seen", "weight": 0.87},
                                       {"text": "ransom payment satisfying divine justice", "weight": 0.84},
                                       {"text": "purging or wiping away defilement", "weight": 0.79}], "confidence": 0.95},
        "forgiveness":  {"root": "סְלִיחָה",  "tr": "selichah","primary": "forgiveness; pardon; divine release from debt",
                          "variants": [{"text": "lifting of guilt and legal penalty", "weight": 0.88},
                                       {"text": "restoration of broken relationship", "weight": 0.85},
                                       {"text": "God's characteristic mercy (Ps 86:5)", "weight": 0.90}], "confidence": 0.95},
        "mercy":        {"root": "חֶסֶד",   "tr": "chesed",  "primary": "steadfast love; covenant loyalty; unfailing kindness",
                          "variants": [{"text": "loyalty within a covenant bond (not random kindness)", "weight": 0.92},
                                       {"text": "God's enduring love that will not let go", "weight": 0.95},
                                       {"text": "compassion rooted in relational commitment", "weight": 0.86}], "confidence": 0.98},
        "grace":        {"root": "חֵן",     "tr": "chen",   "primary": "grace; favour; beauty; unearned goodwill",
                          "variants": [{"text": "favour shown by superior to inferior", "weight": 0.88},
                                       {"text": "aesthetic attractiveness, loveliness", "weight": 0.72},
                                       {"text": "unmerited divine acceptance", "weight": 0.90}], "confidence": 0.94},
        "healing":      {"root": "רְפוּאָה",  "tr": "refuah", "primary": "healing; restoration to wholeness; remedy",
                          "variants": [{"text": "physical cure of bodily illness", "weight": 0.82},
                                       {"text": "divine restoration of the whole person", "weight": 0.88},
                                       {"text": "healing of national wounds after judgment", "weight": 0.76}], "confidence": 0.91},
        "restoration":  {"root": "שְׁאָר",   "tr": "shear",  "primary": "what remains; restoration of what was lost",
                          "variants": [{"text": "return of scattered exiles", "weight": 0.85},
                                       {"text": "renewal of covenant relationship", "weight": 0.88},
                                       {"text": "reversal of the curse of the fall", "weight": 0.79}], "confidence": 0.92},
        "return":       {"root": "שׁוּב",    "tr": "shuv",   "primary": "return; turn back; repent; restore",
                          "variants": [{"text": "physical return from exile", "weight": 0.82},
                                       {"text": "moral and spiritual repentance", "weight": 0.92},
                                       {"text": "God turning back toward his people", "weight": 0.87}], "confidence": 0.96},
        "judgment":     {"root": "מִשְׁפָּט",  "tr": "mishpat","primary": "judgment; justice; right decision; ordinance",
                          "variants": [{"text": "formal legal verdict in court", "weight": 0.84},
                                       {"text": "God's right governance of history", "weight": 0.90},
                                       {"text": "vindication of the oppressed", "weight": 0.80}], "confidence": 0.95},
        "wrath":        {"root": "חֵמָה",   "tr": "chemah",  "primary": "wrath; burning fury; passion aroused by violation",
                          "variants": [{"text": "God's righteous anger at covenant breach", "weight": 0.88},
                                       {"text": "intense heat of divine displeasure", "weight": 0.84},
                                       {"text": "poison — destructive power of wrath", "weight": 0.66}], "confidence": 0.92},
        "justice":      {"root": "צֶדֶק",   "tr": "tsedek",  "primary": "justice; righteousness; right-ordering",
                          "variants": [{"text": "conformity to God's standard of right", "weight": 0.90},
                                       {"text": "legal equity and fair dealing", "weight": 0.85},
                                       {"text": "right relationship in community", "weight": 0.79}], "confidence": 0.96},
        "punishment":   {"root": "עֹנֶשׁ",  "tr": "onesh",  "primary": "punishment; penalty; consequence of evil",
                          "variants": [{"text": "legal penalty for breaking law", "weight": 0.85},
                                       {"text": "disciplinary chastisement for correction", "weight": 0.78},
                                       {"text": "covenant curse enacted", "weight": 0.82}], "confidence": 0.89},
        "fire judgment":{"root": "אֵשׁ מִשְׁפָּט","tr": "esh mishpat","primary": "judgment by fire; consuming divine verdict",
                          "variants": [{"text": "Sodom and Gomorrah — total destruction", "weight": 0.85},
                                       {"text": "purifying fire separating dross from gold", "weight": 0.79},
                                       {"text": "eschatological fire of final judgment", "weight": 0.82}], "confidence": 0.88},
        "flood":        {"root": "מַבּוּל",  "tr": "mabbul", "primary": "flood; deluge; cosmic waters of judgment",
                          "variants": [{"text": "Noahic reset of corrupted creation", "weight": 0.90},
                                       {"text": "uncreation — return to watery chaos", "weight": 0.84},
                                       {"text": "divine judgment followed by covenant", "weight": 0.88}], "confidence": 0.93},
        "plague":       {"root": "מַגֵּפָה",  "tr": "maggephah","primary": "plague; blow; striking defeat",
                          "variants": [{"text": "divine blow against Egypt (Exodus)", "weight": 0.88},
                                       {"text": "outbreak of disease as covenant curse", "weight": 0.82},
                                       {"text": "military defeat as divine judgment", "weight": 0.74}], "confidence": 0.90},
        "exile":        {"root": "גָּלוּת",  "tr": "galut",  "primary": "exile; captivity; diaspora; forced displacement",
                          "variants": [{"text": "covenant curse of land expulsion", "weight": 0.90},
                                       {"text": "Babylonian captivity as divine discipline", "weight": 0.88},
                                       {"text": "spiritual alienation from God's presence", "weight": 0.79}], "confidence": 0.94},
        "destruction":  {"root": "שָׁמַד",   "tr": "shamad", "primary": "destruction; extermination; total ruin",
                          "variants": [{"text": "complete annihilation with no remnant", "weight": 0.88},
                                       {"text": "herem — sacred destruction in holy war", "weight": 0.80},
                                       {"text": "final consequence of unrepented evil", "weight": 0.83}], "confidence": 0.91},
        "remnant":      {"root": "שְׁאָרִית", "tr": "sheerit","primary": "remnant; survivors; the preserved faithful few",
                          "variants": [{"text": "those preserved through catastrophic judgment", "weight": 0.90},
                                       {"text": "kernel of future restoration and hope", "weight": 0.87},
                                       {"text": "the true Israel within ethnic Israel", "weight": 0.80}], "confidence": 0.95},
        "trust":        {"root": "בָּטַח",   "tr": "batach",  "primary": "trust; confident security; reliance",
                          "variants": [{"text": "security that rests and does not fear", "weight": 0.88},
                                       {"text": "foolish overconfidence (negative use)", "weight": 0.65},
                                       {"text": "active leaning upon God's strength", "weight": 0.90}], "confidence": 0.93},
        "fear":         {"root": "יִרְאָה",  "tr": "yirah",  "primary": "fear; reverence; awe; the beginning of wisdom",
                          "variants": [{"text": "trembling dread of divine holiness", "weight": 0.83},
                                       {"text": "worshipful awe that motivates obedience", "weight": 0.92},
                                       {"text": "practical wisdom as daily God-consciousness", "weight": 0.87}], "confidence": 0.96},
        "love":         {"root": "אַהֲבָה",  "tr": "ahavah",  "primary": "love; deep affection; covenantal devotion",
                          "variants": [{"text": "God's initiating elective love for Israel", "weight": 0.92},
                                       {"text": "loyal commitment sustaining relationship", "weight": 0.88},
                                       {"text": "erotic and familial longing (Song of Songs)", "weight": 0.73}], "confidence": 0.97},
        "prayer":       {"root": "תְּפִלָּה", "tr": "tefillah","primary": "prayer; intercession; mediatory address to God",
                          "variants": [{"text": "petitioning God on behalf of another", "weight": 0.85},
                                       {"text": "direct communion with the living God", "weight": 0.90},
                                       {"text": "liturgical prayer as covenant conversation", "weight": 0.78}], "confidence": 0.93},
        "praise":       {"root": "תְּהִלָּה", "tr": "tehillah","primary": "praise; song of glory; boastful acclaim of God",
                          "variants": [{"text": "joyful vocal declaration of God's acts", "weight": 0.88},
                                       {"text": "the content of the Psalter (Tehillim)", "weight": 0.82},
                                       {"text": "God's own renown among the nations", "weight": 0.77}], "confidence": 0.92},
        "worship":      {"root": "שָׁחָה",   "tr": "shachah", "primary": "worship; bow down; prostrate in reverence",
                          "variants": [{"text": "full prostration before divine majesty", "weight": 0.90},
                                       {"text": "total surrender of will to God", "weight": 0.86},
                                       {"text": "cultic ritual of tribute and honour", "weight": 0.74}], "confidence": 0.94},
        "sacrifice":    {"root": "זֶבַח",    "tr": "zevach",  "primary": "sacrifice; slaughtered offering; communal feast",
                          "variants": [{"text": "animal substitution bearing covenant penalty", "weight": 0.88},
                                       {"text": "communal meal shared with God", "weight": 0.77},
                                       {"text": "costly giving expressing devotion", "weight": 0.83}], "confidence": 0.93},
        "seeking":      {"root": "דָּרַשׁ",  "tr": "darash",  "primary": "seek; inquire; require; consult",
                          "variants": [{"text": "diligent investigation of God's ways", "weight": 0.86},
                                       {"text": "oracular inquiry of a prophet or priest", "weight": 0.78},
                                       {"text": "urgent longing pursuit of God himself", "weight": 0.90}], "confidence": 0.92},
        "waiting":      {"root": "קָוָה",    "tr": "qavah",   "primary": "wait; hope; gather; expectant endurance",
                          "variants": [{"text": "active hope that endures against evidence", "weight": 0.90},
                                       {"text": "threads gathered into a cord — strength in tension", "weight": 0.74},
                                       {"text": "eschatological expectation of God's act", "weight": 0.84}], "confidence": 0.93},
        "shepherd":     {"root": "רֹעֶה",   "tr": "ro'eh",   "primary": "shepherd; herdsman; the one who feeds and leads",
                          "variants": [{"text": "God as caring shepherd of his flock", "weight": 0.92},
                                       {"text": "king or leader responsible for the people", "weight": 0.85},
                                       {"text": "pastoral care with rod and staff", "weight": 0.80}], "confidence": 0.95},
    }


def download_old_testament() -> str:
    """Download Hebrew Old Testament text (with KJV fallback)."""

    sources = [
        {
            "name": "Open Scriptures MorphHB archive",
            "url": "https://codeload.github.com/openscriptures/morphhb/zip/refs/tags/v.2.2",
            "loader": _extract_morphhb_text,
        },
        {
            "name": "Door43 unfoldingWord Hebrew Bible archive",
            "url": "https://git.door43.org/unfoldingWord/hbo_uhb/archive/v2.1.31.zip",
            "loader": _extract_door43_text,
        },
    ]

    print("[ Downloading Hebrew Old Testament... ]")

    text = None
    for source in sources:
        try:
            archive_bytes = _download_bytes(source["url"])
            candidate = source["loader"](archive_bytes)
            if len(candidate) > 10000:
                text = candidate
                print(f"[ Loaded from {source['name']} ]")
                break
        except Exception as e:
            print(f"[ Source failed: {e} — trying next... ]")
            continue

    if text is None:
        print("[ Hebrew sources unavailable — falling back to KJV ]")
        return _download_kjv_fallback()

    words = text.split()
    print(f"[ Hebrew OT loaded: {len(words):,} words ]\\n")
    return text


_OLLAMA_MODEL_CACHE: Optional[str] = None
_HEBREW_CHAR_RE = re.compile(r"[\u0590-\u05FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


def _normalize_model_text(text: str) -> str:
    text = text.replace("\r", "\n").strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\"'")


def _sanitize_hebrew_text(text: str) -> str:
    text = _normalize_model_text(text)
    text = re.sub(r"[^\u0590-\u05FF\s\-־׳״.,;:!?()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_refusal(text: str) -> bool:
    lowered = _normalize_model_text(text).lower()
    refusal_markers = [
        "i'm not able",
        "i am not able",
        "i cannot",
        "can't help",
        "can i help you with something else",
        "fulfill your request",
        "translate a biblical hebrew text",
        "sorry",
    ]
    return any(marker in lowered for marker in refusal_markers)


def _is_viable_hebrew_text(text: str) -> bool:
    cleaned = _sanitize_hebrew_text(text)
    hebrew_chars = len(_HEBREW_CHAR_RE.findall(cleaned))
    latin_chars = len(_LATIN_CHAR_RE.findall(text))
    total_non_space = len(re.sub(r"\s+", "", cleaned))
    if hebrew_chars < 8 or total_non_space < 8:
        return False
    if latin_chars > 0:
        return False
    if _looks_like_refusal(text):
        return False
    return True


def _fallback_hebrew_idea(psi: float, resonance: float) -> Tuple[str, str]:
    if psi < 0.08:
        return (
            "אל־תירא כי שלום יבוא אחרי הסערה.",
            "Do not fear, for peace will come after the storm.",
        )
    if resonance > 0.7:
        return (
            "בקשו חכמה ושמרו דרך אמת בלבבכם.",
            "Seek wisdom and keep the way of truth in your heart.",
        )
    return (
        "מה־זאת הרוח בקרבנו ואיך נמצא אור בתוך התהום.",
        "What is this spirit within us, and how shall we find light in the deep?",
    )


def _get_available_ollama_models() -> List[str]:
    req = urllib.request.Request(
        "http://localhost:11434/api/tags",
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return [model.get("name", "") for model in data.get("models", []) if model.get("name")]


def _resolve_ollama_model() -> str:
    global _OLLAMA_MODEL_CACHE
    if _OLLAMA_MODEL_CACHE:
        return _OLLAMA_MODEL_CACHE

    preferred = os.environ.get("OLLAMA_MODEL", "").strip()
    fallback_order = [
        preferred,
        "llama3.2:1b",
        "llama3.1:8b",
        "llama3.2",
        "llama3.1",
    ]
    fallback_order = [name for name in fallback_order if name]

    try:
        available = _get_available_ollama_models()
    except Exception:
        available = []

    for candidate in fallback_order:
        if not available or candidate in available:
            _OLLAMA_MODEL_CACHE = candidate
            return candidate

    if available:
        _OLLAMA_MODEL_CACHE = available[0]
        return _OLLAMA_MODEL_CACHE

    _OLLAMA_MODEL_CACHE = fallback_order[0]
    return _OLLAMA_MODEL_CACHE


def call_local_model(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    candidates = [_resolve_ollama_model(), "llama3.2:1b", "llama3.1:8b"]
    seen = set()

    last_error = None
    for model_name in candidates:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)

        payload = json.dumps({
            "model": model_name,
            "prompt": full_prompt,
            "stream": False
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["response"].strip()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise
        except Exception as exc:
            last_error = exc
            raise

    raise RuntimeError(
        "No working Ollama model found. Install a model with `ollama pull ...` "
        f"or set OLLAMA_MODEL. Last error: {last_error}"
    )


async def call_claude_api(prompt: str) -> str:
    # Compatibility bridge: route existing async caller to local Ollama model.
    return await asyncio.to_thread(call_local_model, prompt)

class HDCVector:
    def __init__(self, dim: int = 1024, rng: Optional[np.random.Generator] = None):
        self.dim = dim
        if rng is None:
            self.vector = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)
        else:
            self.vector = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    
    def bundle(self, other):
        return (self.vector + other.vector) / 2.0
    
    def bind(self, other):
        return self.vector * other.vector
    
    def permute(self, shifts: int = 1):
        v = np.roll(self.vector, shifts)
        obj = HDCVector.__new__(HDCVector)
        obj.dim = self.dim
        obj.vector = v.copy()
        return obj
    
    def similarity(self, other):
        return np.dot(self.vector, other.vector) / self.dim

class SymbolLibrary:
    def __init__(self, dim=1024, rng: Optional[np.random.Generator] = None):
        self.dim = dim
        self.symbols = {
            "TASK": HDCVector(dim, rng=rng),
            "SOLVE": HDCVector(dim, rng=rng),
            "VALID": HDCVector(dim, rng=rng),
            "PSI": HDCVector(dim, rng=rng),
            "NINE_EIGHT": HDCVector(dim, rng=rng),
            "PROMPT": HDCVector(dim, rng=rng),
            "PAYMENT": HDCVector(dim, rng=rng)
        }
    
    def get(self, name: str):
        return self.symbols[name]

class Agent:
    def __init__(self, agent_id: int, dim=1024, rng: Optional[np.random.Generator] = None):
        self.id = agent_id
        self.currency = 10.0
        self.reputation = 1.0
        self.working_memory = HDCVector(dim, rng=rng)
        self.tasks_completed = 0
    
    def work_on_task(self, random_value: float) -> Tuple[float, bool]:
        if self.currency < 0.5:
            return 0.0, False
        success = random_value < (0.65 + 0.3 * self.reputation)
        if success:
            self.tasks_completed += 1
            self.reputation = min(5.0, self.reputation + 0.08)
            return 2.8, True
        else:
            self.reputation = max(0.1, self.reputation - 0.04)
            return 0.0, False

class AxUniverseSim:
    def __init__(self, dim=1024, num_agents=50):
        self.dim = dim
        self.rng = np.random.default_rng()
        self.symbols = SymbolLibrary(dim, rng=self.rng)
        self.num_agents = num_agents

        # Vectorized agent state
        self.agents_reputation = np.ones(num_agents, dtype=np.float32)
        self.agents_currency = np.full(num_agents, 10.0, dtype=np.float32)
        self.agents_memory = self.rng.choice([-1.0, 1.0], size=(num_agents, dim)).astype(np.float32)

        # Optional compatibility list (not used by simulation hot path)
        self.agents: List[Agent] = [Agent(i, dim, rng=self.rng) for i in range(num_agents)]

        self.global_psi = 0.35
        self.A = np.zeros(dim, dtype=np.float32)
        self.step = 0
        self.history_psi: List[float] = []
        self.memory_noise_scale = 0.0065  # reduced mutation in reality-integrity mode
        self.base_memory_noise_scale = self.memory_noise_scale
        self.chaos_amplitude = 0.018
        self.shock_probability = 0.025
        self.shock_strength = 0.12
        self.chaos_state = float(self.rng.random())
        self.toroidal_skip_probability = 0.018
        self.toroidal_max_skip = 11
        self.toroidal_skip_scale = 0.35
        self.pi_universal = float(np.pi)
        self.reality_integrity_mode = True
        self.last_prompt_text = ""
        self.last_stochastic_intensity = 0.0
        self.last_chaos_kick = 0.0
        self.last_oscillation_force = 0.0
        self.last_shock = 0.0
        self.last_dynamic_noise = 0.0
        self.last_toroidal_fraction = 0.0
        self.history_resonance: List[float] = []
        self._earth_cooldown = False
        self._earth_pending = False
        self.cycle_number = 0
        self.cycle_history: List[dict] = []   # stores every attempt
        self.understanding_threshold = 0.40   # starts easy, raises after success
        self.verbose_data_log = True
        self.log_every_n_steps = 5
        self.new_prompt = ""
        self.oscillation_period = 36.0
        self.oscillation_amplitude = 0.030
        self.oscillation_secondary_period = 91.0
        self.oscillation_secondary_amplitude = 0.012
        self.last_idea_vector: Optional[np.ndarray] = None
        self.last_idea_hebrew = ""
        self.last_idea_english = ""
        self.last_prompt_vector = self.symbols.get("PROMPT").vector.copy()
        self.on_data_log: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        self.on_collapse: Optional[Callable[[int, int], None]] = None  # (step, steps_in_collapse)
        self.on_mind_read: Optional[Callable[[list], None]] = None     # (top concepts list)
        self.on_residual: Optional[Callable[[dict], None]] = None       # residual decomposition callback

        # ── Concept library (HDC ↔ meaning dictionary) ─────────────────────
        self.concept_library: dict = {}
        self.concept_matrix: Optional[np.ndarray] = None
        self.concept_labels: List[str] = []
        self.mind_read_every_n: int = 5
        self.last_mind_read: List[dict] = []

        # ── Residual decomposition state ───────────────────────────────────
        self.M_baseline: Optional[np.ndarray] = None
        self.centroid_snapshots: List[np.ndarray] = []
        self.residual_history: List[dict] = []
        self.residual_novelty_threshold: float = 0.35

        # ── Semantic fuzzy layer (meaning scenarios) ───────────────────────
        self.semantic_scenario_rules = {
            "fall_structure": ["transgression", "judgment", "forgiveness"],
            "covenant_crisis": ["covenant", "transgression", "wrath"],
            "redemption_arc": ["sin", "atonement", "salvation"],
            "wisdom_seeking": ["foolishness", "wisdom", "discernment"],
            "exile_return": ["exile", "remnant", "restoration"],
            "creation_tension": ["void", "creation", "order"],
        }
        self.semantic_scenario_threshold: float = 0.10
        self.last_scenario_confidence: dict = {}
        self.last_residual_scenarios: dict = {}

        # ── Hebrew lexicon (root → meanings + variants) ────────────────────
        self.hebrew_lexicon: dict = {}          # populated by load_hebrew_dictionary()

        # ── Entropy-death tracking ──────────────────────────────────────────
        # The universe dies when Ψ stays above the collapse threshold for
        # collapse_patience consecutive ticks without recovering.
        self.collapse_psi_threshold: float = 0.55   # Ψ level that signals terminal entropy
        self.collapse_patience: int = 40             # ticks above threshold before death
        self._collapse_ticks: int = 0               # running counter
        self.collapsed: bool = False                 # final death flag

    def log_data(self, msg: str):
        print(msg)
        if self.on_data_log:
            self.on_data_log(msg)

    def log_response(self, msg: str):
        print(f"Answer: {msg}")
        if self.on_response:
            self.on_response(msg)
    
    def stochastic_update(self):
        # The 'Error' is unregulated noise. Damp it using Phi.
        phi = 1.61803398875
        damping_brake = phi / np.sqrt(2)

        # Calculate distance from the Unity State (1.0)
        u, _ = self.symmetry_measure()
        stability_error = abs(u - 1.0)

        # Forced convergence: higher error => lower allowable noise
        gamma = (0.75 + 0.4 * np.sin(self.step / 12.0)) / (1.0 + stability_error * damping_brake)

        R_n = self.rng.standard_normal(self.dim, dtype=np.float32)
        noise_term = gamma * R_n * (1200.0 / (1 << 31))
        self.A = np.roll(self.A, 1) * 0.92  # one bit of time passing per tick
        self.A += noise_term
        self.last_stochastic_intensity = float(np.mean(np.abs(self.A)))
        return self.last_stochastic_intensity

    def chaotic_drive(self) -> float:
        if self.chaos_amplitude <= 0.0:
            return 0.0
        # Logistic map near full chaos regime
        self.chaos_state = 3.99 * self.chaos_state * (1.0 - self.chaos_state)
        centered = (self.chaos_state - 0.5) * 2.0
        return float(centered * self.chaos_amplitude)

    def apply_toroidal_skip_noise(self):
        # Toroidal transport: some agents "skip" positions in memory space with wrap-around.
        apply_mask = self.rng.random(self.num_agents) < self.toroidal_skip_probability
        if not np.any(apply_mask):
            return 0.0

        idx = np.flatnonzero(apply_mask)
        shifts = self.rng.integers(-self.toroidal_max_skip, self.toroidal_max_skip + 1, size=idx.size)
        shifts = shifts + (shifts == 0)

        for s in np.unique(shifts):
            rows = idx[shifts == s]
            if rows.size == 0:
                continue
            rolled = np.roll(self.agents_memory[rows], int(s), axis=1)
            self.agents_memory[rows] = (1.0 - self.toroidal_skip_scale) * self.agents_memory[rows] + self.toroidal_skip_scale * rolled
        return float(idx.size / self.num_agents)
    
    def symmetry_measure(self):
        # Preserve Pi as a universal invariant while tracking local turbulence.
        _pi = self.pi_universal
        base = 9.0 / 8.0
        effective_u = base - self.global_psi * 0.92
        return effective_u, self.global_psi

    def earth_is_present(self) -> bool:
        u, psi = self.symmetry_measure()
        # Only fires once per stable window, not every tick
        if psi < 0.18 and abs(u - 1.125) < 0.08:
            if not self._earth_cooldown:
                self._earth_cooldown = True
                self.log_data(
                    f"[ Earth detected | step={self.step} | Ψ={psi:.4f} | U={u:.4f} ]"
                )
                return True
        else:
            self._earth_cooldown = False
        return False

    def tick_interval_ms(self) -> int:
        """Universe slows as it approaches Earth conditions."""
        u, psi = self.symmetry_measure()
        # Linear interpolation: fast at high Ψ, slow as Ψ approaches threshold
        if psi > 0.20:
            return 80   # normal cosmic speed
        elif psi > 0.15:
            # Gradual slowdown in the approach window
            t = (0.20 - psi) / 0.05  # 0.0 → 1.0 as psi drops 0.20 → 0.15
            return int(80 + t * 220)  # 80ms → 300ms
        else:
            return 300  # Earth imminent, moving slowly

    @staticmethod
    def _triangular(x: float, a: float, b: float, c: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return float((x - a) / (b - a + 1e-12))
        return float((c - x) / (c - b + 1e-12))

    def fuzzy_memberships(self, u: float, psi: float, resonance: float) -> dict:
        # U proximity to 9/8 target
        near_target = self._triangular(u, 1.04, 1.125, 1.20)
        off_target = min(1.0, abs(u - 1.125) / 0.18)

        # Psi turbulence
        low_turbulence = max(0.0, min(1.0, (0.10 - psi) / 0.10))
        high_turbulence = max(0.0, min(1.0, (psi - 0.02) / 0.20))

        # Resonance quality
        resonance_high = self._triangular(resonance, 0.00, 0.06, 0.16)
        resonance_low = max(0.0, min(1.0, (0.01 - resonance) / 0.03))

        return {
            "near_target": near_target,
            "off_target": off_target,
            "low_turbulence": low_turbulence,
            "high_turbulence": high_turbulence,
            "resonance_high": resonance_high,
            "resonance_low": resonance_low,
        }

    def fuzzy_control(self, u: float, psi: float, resonance: float, work_efficiency: float) -> dict:
        m = self.fuzzy_memberships(u, psi, resonance)

        # Rule set
        # R1: If turbulence high and work low -> increase stabilization pressure
        r1 = min(m["high_turbulence"], 1.0 - work_efficiency)
        # R2: If near target and resonance high -> allow gentle entropy relaxation
        r2 = min(m["near_target"], m["resonance_high"])
        # R3: If off target and resonance low -> increase correction gain
        r3 = min(m["off_target"], m["resonance_low"])

        stabilize_boost = float(np.clip(0.55 * r1 + 0.45 * r3, 0.0, 1.0))
        entropy_relax = float(np.clip(r2, 0.0, 1.0))
        clarity = float(np.clip(0.5 * m["near_target"] + 0.3 * m["resonance_high"] + 0.2 * m["low_turbulence"], 0.0, 1.0))

        return {
            "stabilize_boost": stabilize_boost,
            "entropy_relax": entropy_relax,
            "clarity": clarity,
        }

    def calculate_collective_resonance(self) -> float:
        # Elementwise reduction is more numerically stable than BLAS matmul on some macOS setups.
        similarities = np.sum(self.agents_memory * self.last_prompt_vector[None, :], axis=1, dtype=np.float32) / self.dim
        return float(np.mean(similarities))

    def translate_resonance_to_text(self, resonance: float) -> str:
        if resonance > 0.05:
            return "Strong consensus has formed across workers."
        if resonance > 0.01:
            return "Partial consensus is emerging, but still soft."
        if resonance > -0.01:
            return "The signal is weak and mostly neutral."
        return "Workers are misaligned and the signal is noisy."

    def encode_text_to_hdc(self, text: str) -> np.ndarray:
        # Deterministic HDC encoding from text
        words = text.lower().split()
        acc = np.zeros(self.dim, dtype=np.float32)
        for i, word in enumerate(words):
            word_rng = np.random.default_rng(seed=hash(word) % (2**32))
            v = word_rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
            acc += np.roll(v, i)  # position encoding via permutation
        return np.sign(acc + 1e-9)

    def _build_concept_library(self):
        concepts = {
            "beginning": "in the beginning",
            "creation": "god created the heavens and the earth",
            "light": "let there be light",
            "darkness": "darkness over the deep",
            "void": "formless and empty void",
            "order": "god separated the light from darkness",
            "firmament": "god made the firmament",
            "earth formed": "dry land appeared",
            "waters": "gathering of the waters",
            "time": "evening and morning the first day",
            "god": "the lord god almighty",
            "spirit": "spirit of god moved upon the waters",
            "word": "the word of the lord came",
            "glory": "the glory of the lord filled the temple",
            "holy": "holy holy holy is the lord of hosts",
            "presence": "walking in the presence of god",
            "angel": "the angel of the lord appeared",
            "fire": "god appeared in fire and cloud",
            "throne": "god sits upon the throne",
            "name": "the name of the lord is holy",
            "covenant": "covenant between god and man",
            "commandment": "keep my commandments",
            "law": "the law of moses",
            "promise": "god promised to abraham",
            "sign": "sign of the covenant",
            "blood": "blood of the covenant",
            "sabbath": "remember the sabbath day",
            "circumcision": "sign of circumcision",
            "obedience": "obey the voice of the lord",
            "transgression": "transgressed the covenant",
            "man": "god formed man from the dust",
            "woman": "woman bone of my bone",
            "breath": "breath of life into his nostrils",
            "soul": "living soul",
            "heart": "the heart of man",
            "flesh": "all flesh",
            "sin": "sin crouching at the door",
            "death": "dust you shall return",
            "birth": "born of woman",
            "suffering": "pain and suffering",
            "wisdom": "wisdom and understanding",
            "knowledge": "tree of knowledge of good and evil",
            "understanding": "give your servant an understanding heart",
            "truth": "the truth of the lord endures",
            "foolishness": "the fool says in his heart",
            "discernment": "discern between good and evil",
            "word of god": "every word that proceeds from god",
            "teaching": "teach them the statutes",
            "counsel": "seek counsel from the lord",
            "mystery": "secret things belong to the lord",
            "salvation": "the lord is my salvation",
            "redemption": "redeemed from the hand of the enemy",
            "deliverance": "deliver us from evil",
            "atonement": "make atonement for sin",
            "forgiveness": "forgive the iniquity of this people",
            "mercy": "steadfast love and mercy",
            "grace": "grace and truth",
            "healing": "he heals the brokenhearted",
            "restoration": "restore the years the locust ate",
            "return": "return to the lord your god",
            "judgment": "the lord rises to judge",
            "wrath": "the wrath of god",
            "justice": "justice and righteousness",
            "punishment": "visited iniquity upon the children",
            "fire judgment": "consuming fire of judgment",
            "flood": "waters covered the earth",
            "plague": "plague upon egypt",
            "exile": "carried into exile",
            "destruction": "utterly destroy",
            "remnant": "a remnant shall return",
            "trust": "trust in the lord with all your heart",
            "fear": "fear of the lord is wisdom",
            "love": "love the lord your god",
            "prayer": "called upon the name of the lord",
            "praise": "praise the lord",
            "worship": "bow down and worship",
            "sacrifice": "offered a sacrifice to the lord",
            "seeking": "seek the lord while he may be found",
            "waiting": "wait upon the lord",
            "shepherd": "the lord is my shepherd",
        }

        self.log_data(f"[ Building concept library — {len(concepts)} concepts ]")
        library = {}
        for label, phrase in concepts.items():
            vector = self.encode_text_to_hdc(phrase).astype(np.float32)
            norm = np.linalg.norm(vector)
            library[label] = vector / norm if norm > 0 else vector

        self.concept_library = library
        self.concept_labels = list(library.keys())
        self.concept_matrix = np.stack([library[label] for label in self.concept_labels], axis=0).astype(np.float32)
        self.log_data(f"[ Concept library ready — {len(self.concept_labels)} entries ]\n")

    # ── Hebrew lexicon ──────────────────────────────────────────────────────

    def load_hebrew_dictionary(self):
        """
        Load the Hebrew root lexicon into self.hebrew_lexicon.

        Strategy (in order of preference):
          1. Read from local cache  ~/.cache/axuniverse/hebrew_lexicon.json
          2. Fall back to the embedded public-domain lexicon immediately
          3. (Optional) background thread: attempt to fetch openscriptures
             lexicon and update the cache for next run.

        The lexicon maps every concept label used in self.concept_library to:
            {root, tr, primary, variants: [{text, weight}], confidence}
        """
        cache_path = os.path.expanduser("~/.cache/axuniverse/hebrew_lexicon.json")

        # 1 — try disk cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and len(data) >= 40:
                    self.hebrew_lexicon = data
                    self.log_data(
                        f"[ Hebrew lexicon loaded from cache — {len(data)} roots ]"
                    )
                    return
            except Exception:
                pass  # corrupt cache — fall through

        # 2 — embedded lexicon (always available, no network needed)
        self.hebrew_lexicon = _build_embedded_hebrew_lexicon()
        self.log_data(
            f"[ Hebrew lexicon ready (embedded) — {len(self.hebrew_lexicon)} roots ]"
        )

        # 3 — write to cache for future runs
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(self.hebrew_lexicon, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass  # non-fatal

    def enrich_concept(self, concept_label: str) -> Optional[str]:
        """
        Return a rich scholarly gloss for a concept label using the loaded
        Hebrew lexicon.  Returns None when the label is not in the lexicon or
        the lexicon is empty.

        Output format (mirrors user spec):
            "[Hebrew: חָכְמָה | chokmah] wisdom — skill, experience…
             (or: moral excellence, divine gift, fear-rooted knowledge)"
            Confidence: 0.97
        """
        if not self.hebrew_lexicon:
            return None
        entry = self.hebrew_lexicon.get(concept_label)
        if entry is None:
            return None

        root        = entry.get("root", "")
        tr          = entry.get("tr", "")
        primary     = entry.get("primary", concept_label)
        variants    = entry.get("variants", [])
        confidence  = float(entry.get("confidence", 1.0))

        variant_texts = [v["text"] for v in variants if v.get("weight", 0) >= 0.65]
        if variant_texts:
            amplified = f"{primary} (or: {', '.join(variant_texts)})"
        else:
            amplified = primary

        header = ""
        if root and tr:
            header = f"[Hebrew: {root} | {tr}] "

        return f"{header}{amplified}  [conf {confidence:.2f}]"

    def read_population_mind(self, top_n: int = 5) -> List[dict]:
        if self.concept_matrix is None or len(self.concept_labels) == 0:
            return []


        centroid = np.mean(self.agents_memory, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm < 1e-9:
            return []
        centroid_unit = centroid / norm

        scores = self.concept_matrix @ centroid_unit

        def _strength(score: float) -> str:
            magnitude = abs(score)
            if magnitude < 0.01:
                return "trace"
            if magnitude < 0.03:
                return "weak"
            if magnitude < 0.06:
                return "moderate"
            if magnitude < 0.10:
                return "strong"
            return "dominant"

        pos_idx = np.where(scores > 0)[0]
        neg_idx = np.where(scores < 0)[0]

        top_pos = pos_idx[np.argsort(scores[pos_idx])[::-1]][:top_n]
        top_neg = neg_idx[np.argsort(np.abs(scores[neg_idx]))[::-1]][:top_n]

        results = []
        for idx in top_pos:
            score = float(scores[idx])
            results.append({
                "concept": self.concept_labels[idx],
                "score": score,
                "strength": _strength(score),
                "polarity": "+",
            })

        for idx in top_neg:
            score = float(scores[idx])
            results.append({
                "concept": self.concept_labels[idx],
                "score": score,
                "strength": _strength(score),
                "polarity": "−",
            })

        results = results[:top_n]

        self.last_mind_read = results
        return results

    def format_mind_read(self, results: List[dict]) -> str:
        if not results:
            return "[ Mind: library not built yet ]"
        parts = [f"{item['polarity']}{item['concept']} ({item['strength']})" for item in results]
        return "[ Mind: " + " · ".join(parts) + " ]"

    def _concept_presence_membership(self, beta: float) -> float:
        return float(np.clip(self._triangular(abs(beta), 0.005, 0.06, 0.18), 0.0, 1.0))

    def _concept_beta_scores(self, vector: np.ndarray) -> dict:
        if self.concept_matrix is None or len(self.concept_labels) == 0:
            return {}
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return {}
        unit = (vector / norm).astype(np.float32)
        scores = self.concept_matrix @ unit
        return {label: float(scores[i]) for i, label in enumerate(self.concept_labels)}

    def tension_membership(self, concept_a: str, concept_b: str, beta_map: dict) -> float:
        beta_a = float(beta_map.get(concept_a, 0.0))
        beta_b = float(beta_map.get(concept_b, 0.0))
        co_presence = min(abs(beta_a), abs(beta_b))

        vec_a = self.concept_library.get(concept_a)
        vec_b = self.concept_library.get(concept_b)
        if vec_a is None or vec_b is None:
            opposition = 0.0
        else:
            similarity = float(np.dot(vec_a, vec_b))
            opposition = float(np.clip(1.0 - max(0.0, similarity), 0.0, 1.0))

        return float(np.clip(co_presence * opposition, 0.0, 1.0))

    def evaluate_scenarios_from_beta_map(self, beta_map: dict) -> dict:
        if not beta_map:
            return {}

        scenario_scores = {}
        for scenario, concepts in self.semantic_scenario_rules.items():
            memberships = [self._concept_presence_membership(beta_map.get(c, 0.0)) for c in concepts]
            base_strength = min(memberships) if memberships else 0.0

            tensions = []
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    tensions.append(self.tension_membership(concepts[i], concepts[j], beta_map))
            tension_strength = float(np.mean(tensions)) if tensions else 0.0

            firing = float(np.clip(base_strength * (0.65 + 0.35 * tension_strength), 0.0, 1.0))
            scenario_scores[scenario] = firing

        return scenario_scores

    def scenario_vector_string(self, scenario_scores: dict, top_n: int = 6) -> str:
        if not scenario_scores:
            return "[ Scenario: unavailable ]"
        ranked = sorted(scenario_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        parts = [f"{name}:{score:.2f}" for name, score in ranked]
        return "[ Scenario: " + " | ".join(parts) + " ]"

    def evaluate_population_scenarios(self) -> dict:
        centroid = np.mean(self.agents_memory, axis=0).astype(np.float32)
        beta_map = self._concept_beta_scores(centroid)
        scenario_scores = self.evaluate_scenarios_from_beta_map(beta_map)
        self.last_scenario_confidence = scenario_scores

        if not scenario_scores:
            return {
                "scenario_label": "unknown",
                "meaning_weight": 0.0,
                "unresolved_fraction": 1.0,
                "scores": {},
            }

        scenario_label, meaning_weight = max(scenario_scores.items(), key=lambda kv: kv[1])
        unresolved_fraction = float(np.clip(1.0 - meaning_weight, 0.0, 1.0))
        return {
            "scenario_label": scenario_label,
            "meaning_weight": float(meaning_weight),
            "unresolved_fraction": unresolved_fraction,
            "scores": scenario_scores,
            "beta_map": beta_map,
        }

    def snapshot_centroid(self) -> np.ndarray:
        centroid = np.mean(self.agents_memory, axis=0).astype(np.float32).copy()
        self.centroid_snapshots.append(centroid)
        return centroid

    def interpret_residual(self, residual_vec: np.ndarray) -> dict:
        beta_map = self._concept_beta_scores(residual_vec)
        scenario_scores = self.evaluate_scenarios_from_beta_map(beta_map)
        self.last_residual_scenarios = scenario_scores
        return scenario_scores

    def compute_residual(self, label: str = "") -> dict:
        if self.concept_matrix is None or self.M_baseline is None:
            return {}

        current = np.mean(self.agents_memory, axis=0).astype(np.float32)
        delta = current - self.M_baseline
        delta_norm = float(np.linalg.norm(delta))

        if delta_norm < 1e-9:
            result = {
                "cycle": self.cycle_number,
                "label": label,
                "delta_norm": 0.0,
                "decomposition": [],
                "residual_norm": 0.0,
                "named_norm": 0.0,
                "coverage": 0.0,
                "novel_flag": False,
                "top_residual_concepts": [],
                "residual_scenarios": {},
            }
            self.residual_history.append(result)
            if self.on_residual:
                self.on_residual(result)
            return result

        delta_unit = (delta / delta_norm).astype(np.float32)
        betas = (self.concept_matrix @ delta_unit).astype(np.float32)

        named_part = np.zeros(self.dim, dtype=np.float32)
        for i, beta in enumerate(betas):
            named_part += float(beta) * self.concept_matrix[i]

        residual = delta_unit - named_part
        residual_norm = float(np.linalg.norm(residual))
        named_norm = float(np.linalg.norm(named_part))
        coverage = float(np.clip(named_norm / (delta_norm + 1e-9), 0.0, 1.0))

        def _strength(beta: float) -> str:
            mag = abs(beta)
            if mag < 0.01:
                return "trace"
            if mag < 0.03:
                return "weak"
            if mag < 0.06:
                return "moderate"
            if mag < 0.10:
                return "strong"
            return "dominant"

        sorted_idx = np.argsort(np.abs(betas))[::-1][:8]
        decomposition = [
            {
                "concept": self.concept_labels[i],
                "beta": float(betas[i]),
                "strength": _strength(float(betas[i])),
                "polarity": "+" if betas[i] >= 0 else "−",
            }
            for i in sorted_idx
        ]

        top_residual_concepts = []
        r_norm = float(np.linalg.norm(residual))
        if r_norm > 1e-9:
            r_unit = (residual / r_norm).astype(np.float32)
            r_scores = self.concept_matrix @ r_unit
            top_r_idx = np.argsort(np.abs(r_scores))[::-1][:3]
            top_residual_concepts = [
                {"concept": self.concept_labels[i], "score": float(r_scores[i])}
                for i in top_r_idx
            ]

        residual_scenarios = self.interpret_residual(residual)
        novel_flag = residual_norm > self.residual_novelty_threshold

        result = {
            "cycle": self.cycle_number,
            "label": label,
            "delta_norm": delta_norm,
            "decomposition": decomposition,
            "residual_norm": residual_norm,
            "named_norm": named_norm,
            "coverage": coverage,
            "novel_flag": novel_flag,
            "top_residual_concepts": top_residual_concepts,
            "residual_scenarios": residual_scenarios,
        }
        self.residual_history.append(result)

        if novel_flag:
            self.log_data(
                f"[ ◈ RESIDUAL NOVELTY — cycle {self.cycle_number} | ||R||={residual_norm:.4f} | coverage={coverage*100:.1f}% ]"
            )
        else:
            self.log_data(
                f"[ ◇ Residual — cycle {self.cycle_number} | ||R||={residual_norm:.4f} | coverage={coverage*100:.1f}% ]"
            )

        if residual_scenarios:
            self.log_data("[ Residual scenarios ] " + self.scenario_vector_string(residual_scenarios, top_n=4))

        if self.on_residual:
            self.on_residual(result)

        return result

    def inject_concept(self, concept_label: str, strength: float = 1.0) -> dict:
        if concept_label not in self.concept_library:
            matches = [label for label in self.concept_library if concept_label.lower() in label]
            if not matches:
                self.log_data(f"[ inject_concept: '{concept_label}' not in library ]")
                return {"error": "concept not found"}
            concept_label = matches[0]

        concept_vec = self.concept_library[concept_label]
        carrier = np.sign(self.A)
        if np.count_nonzero(carrier) < int(0.05 * self.dim):
            carrier = np.ones(self.dim, dtype=np.float32)
        bound = concept_vec * carrier

        pre_similarities = np.dot(self.agents_memory, bound) / self.dim
        pre_resonance = float(np.mean(pre_similarities))

        k = max(1, int(np.ceil(0.60 * self.num_agents)))
        top_idx = np.argpartition(pre_similarities, -k)[-k:]
        alignment_mask = np.zeros(self.num_agents, dtype=bool)
        alignment_mask[top_idx] = True

        mixed = (
            (1.0 - strength * 0.4) * self.agents_memory[alignment_mask]
            + (strength * 0.4) * bound[None, :]
        )
        self.agents_memory[alignment_mask] = np.sign(mixed).astype(np.float32)
        self.agents_memory[self.agents_memory == 0.0] = 1.0

        self.agents_currency[alignment_mask] += 20.0 * strength
        self.agents_reputation[alignment_mask] = np.minimum(
            5.0, self.agents_reputation[alignment_mask] + 0.4 * strength
        )

        self.global_psi *= 0.5 + 0.2 * strength
        self.last_prompt_vector = bound.copy()

        post_similarities = np.dot(self.agents_memory, bound) / self.dim
        post_resonance = float(np.mean(post_similarities))

        non_aligned_mask = ~alignment_mask
        if np.any(non_aligned_mask):
            noise = self.rng.standard_normal(self.dim).astype(np.float32)
            noise -= np.dot(noise, concept_vec) * concept_vec
            noise_norm = np.linalg.norm(noise)
            if noise_norm > 1e-9:
                noise = noise / noise_norm
            drift_scale = 0.15 * strength

            rows = self.agents_memory[non_aligned_mask]
            jitter = self.rng.standard_normal((rows.shape[0], self.dim)).astype(np.float32) * 0.05
            rows = np.sign(rows + drift_scale * noise[None, :] + jitter).astype(np.float32)
            rows[rows == 0.0] = 1.0
            self.agents_memory[non_aligned_mask] = rows

        aligned = int(alignment_mask.sum())
        self.log_data(
            f"[ ↓ Injected concept '{concept_label}' | resonance {pre_resonance:+.4f} → {post_resonance:+.4f} | "
            f"{aligned}/{self.num_agents} agents aligned | Ψ → {self.global_psi:.4f} ]"
        )
        return {
            "concept": concept_label,
            "pre_resonance": pre_resonance,
            "post_resonance": post_resonance,
            "aligned": aligned,
            "psi": self.global_psi,
        }

    def measure_reception(self, idea_vector: np.ndarray) -> dict:
        """Measure how well the idea propagated through the agent population."""

        # 1. Per-agent similarity to the idea
        similarities = np.dot(self.agents_memory, idea_vector) / self.dim

        # 2. Population split — threshold at 0.02 (weak but present signal)
        understood_mask = similarities > 0.02
        understood_count = int(np.sum(understood_mask))
        missed_count = self.num_agents - understood_count

        # 3. Resonance delta — compare to baseline random (~0.0)
        resonance = float(np.mean(similarities))

        # 4. Comprehension quality — did high-rep agents understand better?
        if understood_count > 0:
            understood_rep = float(np.mean(
                self.agents_reputation[understood_mask]))
            all_rep = float(np.mean(self.agents_reputation))
            comprehension_quality = understood_rep / max(all_rep, 0.01)
        else:
            comprehension_quality = 0.0

        # 5. Ψ responsiveness — how much did universe react
        u, psi = self.symmetry_measure()

        return {
            "understood": understood_count,
            "missed": missed_count,
            "total": self.num_agents,
            "resonance": resonance,
            "comprehension_quality": comprehension_quality,
            "psi": psi,
            "similarities": similarities.tolist(),
        }

    def ingest_response(self, person_response: str) -> int:
        """Inject known concepts directly; otherwise encode the raw response as HDC."""

        response_lower = person_response.lower().strip()
        matched_concept = None
        if response_lower in self.concept_library:
            matched_concept = response_lower
        else:
            for label in self.concept_labels:
                if response_lower in label or label in response_lower:
                    matched_concept = label
                    break

        if matched_concept:
            result = self.inject_concept(matched_concept, strength=1.0)
            return int(result.get("aligned", 0))

        self.log_data(f"[ Response '{person_response[:40]}' — encoding directly as HDC (no Ollama) ]")
        response_vec = self.encode_text_to_hdc(person_response)
        carrier = np.sign(self.A)
        if np.count_nonzero(carrier) < int(0.05 * self.dim):
            carrier = np.sign(self.last_prompt_vector)
        if np.count_nonzero(carrier) < int(0.05 * self.dim):
            carrier = np.ones(self.dim, dtype=np.float32)

        bound = response_vec * carrier
        similarities = np.dot(self.agents_memory, bound) / self.dim

        # Keep a stable fraction of the population aligned even near deep-stable Ψ.
        k = max(1, int(np.ceil(0.60 * self.num_agents)))
        top_idx = np.argpartition(similarities, -k)[-k:]
        alignment_mask = np.zeros(self.num_agents, dtype=bool)
        alignment_mask[top_idx] = True

        self.agents_currency[alignment_mask] += 25.0
        self.agents_reputation[alignment_mask] = np.minimum(
            5.0, self.agents_reputation[alignment_mask] + 0.5
        )
        self.global_psi *= 0.6
        self.last_prompt_vector = bound.copy()

        aligned = int(alignment_mask.sum())
        self.log_data(f"[ {aligned}/{self.num_agents} agents amplified. Ψ → {self.global_psi:.4f} ]")
        return aligned

    def earth_response(self, prompt: str) -> Optional[str]:
        p = prompt.lower().strip()

        if "pi" in p:
            return f"Earth response: Pi is a universal constant, approximately {self.pi_universal:.15f}."

        if "how many sides" in p and "square" in p:
            return "Earth response: A square has 4 sides."

        if "earth" in p and any(k in p for k in ["where", "find", "locate"]):
            return "Earth response: Earth is the communication anchor in this model. Route prompt through the Earth channel and return the direct semantic answer."

        # Lightweight arithmetic parsing
        tokens = p.replace("?", " ").split()
        if len(tokens) >= 3:
            try:
                for i in range(len(tokens) - 2):
                    a = float(tokens[i])
                    op = tokens[i + 1]
                    b = float(tokens[i + 2])
                    if op in ["+", "plus"]:
                        return f"Earth response: {a + b:g}"
                    if op in ["-", "minus"]:
                        return f"Earth response: {a - b:g}"
                    if op in ["*", "x", "times"]:
                        return f"Earth response: {a * b:g}"
                    if op in ["/", "div", "divide", "divided"] and b != 0:
                        return f"Earth response: {a / b:g}"
            except ValueError:
                pass

        return None

    def parse_noise_aspects(self, resonance: float) -> dict:
        # Normalize current noise channels to [0, 1] for comparability.
        chaos_n = float(np.clip(abs(self.last_chaos_kick) / max(self.chaos_amplitude, 1e-9), 0.0, 1.0))
        shock_n = float(np.clip(abs(self.last_shock) / max(self.shock_strength, 1e-9), 0.0, 1.0))
        mut_n = float(np.clip(self.last_dynamic_noise / max(self.base_memory_noise_scale * 1.6, 1e-9), 0.0, 1.0))
        tor_n = float(np.clip(self.last_toroidal_fraction / max(self.toroidal_skip_probability, 1e-9), 0.0, 1.0))
        sto_n = float(np.clip(self.last_stochastic_intensity / 0.0012, 0.0, 1.0))

        channels = {
            "chaos": chaos_n,
            "shock": shock_n,
            "mutation": mut_n,
            "toroidal": tor_n,
            "stochastic": sto_n,
        }
        ranked = sorted(channels.items(), key=lambda kv: kv[1], reverse=True)
        dominant = [name for name, score in ranked if score > 0.45][:2]
        if not dominant:
            dominant = [ranked[0][0]]

        # Estimate effective signal quality from resonance and aggregate noise pressure.
        noise_pressure = float(np.mean(list(channels.values())))
        signal_quality = float(np.clip((resonance + 1.0) * 0.5 * (1.0 - 0.7 * noise_pressure), 0.0, 1.0))

        hint_map = {
            "chaos": "increase temporal averaging",
            "shock": "wait a few ticks after spikes",
            "mutation": "reduce identity mutation or increase workers",
            "toroidal": "lower skip probability/scale for locality",
            "stochastic": "raise damping to suppress random drift",
        }
        hints = [hint_map[d] for d in dominant]

        return {
            "channels": channels,
            "dominant": dominant,
            "noise_pressure": noise_pressure,
            "signal_quality": signal_quality,
            "hints": hints,
        }

    def generate_answer(self, prompt: str) -> str:
        if self.reality_integrity_mode:
            earth = self.earth_response(prompt)
            if earth is not None:
                return earth

        u, psi = self.symmetry_measure()
        resonance = self.calculate_collective_resonance()
        fuzzy = self.fuzzy_control(u=u, psi=psi, resonance=resonance, work_efficiency=0.5)
        clarity = fuzzy["clarity"]
        noise = self.parse_noise_aspects(resonance=resonance)
        adjusted_clarity = float(np.clip(clarity * (1.0 - 0.55 * noise["noise_pressure"]) + 0.35 * noise["signal_quality"], 0.0, 1.0))
        prompt_l = prompt.lower().strip()

        if any(k in prompt_l for k in ["noise", "chaos", "toroid", "turbulence", "signal"]):
            dom = ", ".join(noise["dominant"])
            hints = "; ".join(noise["hints"])
            return (
                f"Noise parse: dominant channels = {dom}. "
                f"Estimated signal quality = {noise['signal_quality']:.2f}, effective clarity = {adjusted_clarity:.2f}. "
                f"To get closer to the prompt answer: {hints}. "
                f"Current interpretation: {self.translate_resonance_to_text(resonance)}"
            )

        if abs(u - 1.125) < 0.05 and adjusted_clarity > 0.45:
            return f"The universe has crystallized a response: {self.translate_resonance_to_text(resonance)}"
        if adjusted_clarity > 0.30:
            return "The universe is partially coherent. A weak answer is forming, but uncertainty is still high."
        return "The universe is too turbulent to provide a clear answer. More work is required."

    def form_idea(self) -> str:
        """Derive transmission purely from population geometry (no Ollama)."""
        self.cycle_number += 1
        _, psi = self.symmetry_measure()

        mind = self.read_population_mind(top_n=8)
        positives = [item for item in mind if item["polarity"] == "+"]
        negatives = [item for item in mind if item["polarity"] == "−"]

        if positives:
            primary = positives[0]
        elif mind:
            primary = mind[0]
        else:
            primary = {"concept": "covenant", "score": 0.0, "strength": "trace", "polarity": "+"}

        top_concept = primary["concept"]
        top_score = primary["score"]
        supporting = [item["concept"] for item in positives[1:3]]
        opposed = [item["concept"] for item in negatives[:2]]

        self.last_idea_vector = self.concept_library.get(
            top_concept, self.encode_text_to_hdc(top_concept)
        ).copy()
        self.last_prompt_vector = self.last_idea_vector.copy()
        self.last_idea_hebrew = ""

        strength_phrase = {
            "dominant": "overwhelmingly",
            "strong": "strongly",
            "moderate": "clearly",
            "weak": "faintly",
            "trace": "barely",
        }.get(primary["strength"], "")

        if top_score > 0:
            lines = [f"The agents are {strength_phrase} aligned with: {top_concept.upper()}."]
        else:
            lines = [f"The agents are {strength_phrase} opposed to: {top_concept.upper()}."]

        if supporting:
            lines.append(f"Supported by: {' · '.join(name.upper() for name in supporting)}.")
        if opposed:
            lines.append(f"They reject: {' · '.join(name.upper() for name in opposed)}.")

        if psi < 0.10:
            lines.append(f"Universe is calm (Ψ={psi:.3f}). Signal is clear.")
        elif psi < 0.30:
            lines.append(f"Universe is active (Ψ={psi:.3f}). Signal is readable.")
        else:
            lines.append(f"Universe is turbulent (Ψ={psi:.3f}). Signal is strained.")

        idea_english = " ".join(lines)

        # ── Hebrew lexicon enrichment (reality_integrity_mode) ─────────────
        if self.reality_integrity_mode and self.hebrew_lexicon:
            gloss = self.enrich_concept(top_concept)
            if gloss:
                idea_english = idea_english + f"\n  ↳ {gloss}"
                # log supporting concepts too (confidence ≥ 0.90)
                support_glosses = []
                for sc in supporting:
                    entry = self.hebrew_lexicon.get(sc)
                    if entry and float(entry.get("confidence", 0)) >= 0.90:
                        sg = self.enrich_concept(sc)
                        if sg:
                            support_glosses.append(f"  ↳ ({sc}) {sg}")
                if support_glosses:
                    idea_english += "\n" + "\n".join(support_glosses)

        self.last_idea_english = idea_english

        self.log_data(
            f"[ Cycle {self.cycle_number} | PRIMARY: {top_concept.upper()} ({top_score:+.4f}) | "
            f"supports: {supporting} | opposes: {opposed} | Ψ={psi:.3f} ]"
        )

        return idea_english

    def record_cycle(self, idea_english: str, reception: dict, response: str = ""):
        understood_pct = reception["understood"] / reception["total"] * 100

        if reception["resonance"] > 0.05:
            signal_quality = "STRONG"
        elif reception["resonance"] > 0.01:
            signal_quality = "PARTIAL"
        elif reception["resonance"] > -0.01:
            signal_quality = "WEAK"
        else:
            signal_quality = "NOISY"

        self.cycle_history.append({
            "cycle": self.cycle_number,
            "idea_hebrew": getattr(self, "last_idea_hebrew", ""),
            "idea_english": idea_english,
            "understood_pct": understood_pct,
            "signal_quality": signal_quality,
            "response": response,
            "psi_at_transmission": reception["psi"],
        })

        success = understood_pct >= (self.understanding_threshold * 100)

        if success:
            self.understanding_threshold = min(0.85, self.understanding_threshold + 0.05)
            self.log_data(
                f"[ Cycle {self.cycle_number} UNDERSTOOD — "
                f"{understood_pct:.0f}% reception | "
                f"threshold → {self.understanding_threshold*100:.0f}% ]"
            )
        else:
            self.log_data(
                f"[ Cycle {self.cycle_number} FAILED — "
                f"{understood_pct:.0f}% < "
                f"{self.understanding_threshold*100:.0f}% needed ]"
            )

        self.compute_residual(label=response)

        return success

    def seed_from_text(self, text: str, weight: float = 0.35):
        """Encode a text corpus into agent memories before the sim starts."""
        words = text.split()
        chunk_size = 20
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        self.log_data(
            f"[ Seeding {self.num_agents} agents from {len(chunks)} "
            f"chunks ({len(words):,} words) ]"
        )
        for agent_idx in range(self.num_agents):
            sample_size = max(1, min(len(chunks), len(chunks) // self.num_agents + 50))
            sampled_idx = self.rng.choice(len(chunks), size=sample_size, replace=False)
            acc = np.zeros(self.dim, dtype=np.float32)
            for ci in sampled_idx:
                acc += self.encode_text_to_hdc(chunks[ci])
            text_memory = np.sign(acc + 1e-9).astype(np.float32)
            self.agents_memory[agent_idx] = np.sign(
                (1.0 - weight) * self.agents_memory[agent_idx]
                + weight * text_memory + 1e-9
            ).astype(np.float32)

        key_phrases = [
            "in the beginning god created",
            "the lord is my shepherd",
            "wisdom and understanding",
            "covenant between god and man",
            "the word of the lord",
        ]
        for phrase in key_phrases:
            v = self.encode_text_to_hdc(phrase)
            self.last_prompt_vector = np.sign(
                0.8 * self.last_prompt_vector + 0.2 * v + 1e-9
            ).astype(np.float32)

        self.log_data("[ Universe seeded. Agents carry the scripture. ]")
        self._build_concept_library()
        self.load_hebrew_dictionary()
        self.M_baseline = np.mean(self.agents_memory, axis=0).astype(np.float32).copy()

    def run_with_observer(self):
        print("=== Universe running. Waiting for Earth... ===\n")
        while True:
            self.run_tick()

            if self.earth_is_present():
                print("\n[ Earth has appeared ]")
                idea = self.form_idea()

                print(f"\n  IDEA → {idea}")
                print("\n  Your response: ", end="", flush=True)
                response = input()

                if response.strip():
                    self.ingest_response(response)
                    print("[ Amplified. Universe continues. ]\n")
    
    def broadcast_prompt(self):
        prompt_vec = self.symbols.get("PROMPT").vector

        asked_prompt = None
        if self.new_prompt.strip():
            asked_prompt = self.new_prompt.strip()
            self.last_prompt_text = asked_prompt
            self.last_prompt_vector = prompt_vec.copy()
            self.log_data(f"Broadcasting custom prompt: {asked_prompt[:40]}...")
            self.new_prompt = ""

        # Vectorized memory broadcast
        self.agents_memory += prompt_vec[None, :]
        self.agents_memory *= 0.5

        resonance_probe = abs(self.calculate_collective_resonance())
        coherence_lock = max(0.0, resonance_probe - 0.92)

        # Reduced randomness in reality-integrity mode, but never allow perfect lock.
        dynamic_noise = self.base_memory_noise_scale * (
            0.9
            + 1.4 * abs(self.last_chaos_kick)
            + 0.8 * self.global_psi
            + 1.8 * coherence_lock
        )
        self.last_dynamic_noise = float(dynamic_noise)
        mutation = self.rng.standard_normal((self.num_agents, self.dim), dtype=np.float32) * dynamic_noise
        self.agents_memory += mutation

        # Toroidal, wrap-around skip noise adds non-local jumps along the vector manifold.
        self.last_toroidal_fraction = self.apply_toroidal_skip_noise()

        # Tiny drift prevents perfect resonance lock
        self.agents_memory += self.rng.standard_normal(
            (self.num_agents, self.dim)).astype(np.float32) * (0.002 + 0.004 * coherence_lock)

        # Concept-orthogonal drift to avoid concept lock-in.
        if self.concept_matrix is not None and self.step % 3 == 0:
            axis = self.last_prompt_vector.astype(np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-9:
                axis_unit = axis / axis_norm
                perp = self.rng.standard_normal(self.dim).astype(np.float32)
                perp -= np.dot(perp, axis_unit) * axis_unit
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-9:
                    perp = perp / perp_norm * 0.008
                    self.agents_memory += perp[None, :]

        # Re-normalize to maintain HDC integrity
        np.sign(self.agents_memory, out=self.agents_memory)
        self.agents_memory[self.agents_memory == 0.0] = 1.0

        if asked_prompt is not None:
            self.log_response(self.generate_answer(asked_prompt))
    
    def run_tick(self):
        if self.collapsed:
            return  # universe is dead — no further simulation

        self.step += 1

        # 1) Natural entropy forces ongoing maintenance work
        age_entropy = min(0.018, self.step * 0.00005)
        cycle_entropy = min(0.020, len(self.cycle_history) * 0.0025)
        low_psi_penalty = max(0.0, 0.03 - self.global_psi) * 0.8
        self.global_psi += 0.010 + age_entropy + cycle_entropy + low_psi_penalty

        oscillation_force = (
            self.oscillation_amplitude * np.sin((2.0 * np.pi * self.step) / self.oscillation_period)
            + self.oscillation_secondary_amplitude
            * np.sin((2.0 * np.pi * self.step) / self.oscillation_secondary_period + 0.45 * self.cycle_number)
        )
        if self.global_psi < 0.08:
            oscillation_force += 0.020
        self.last_oscillation_force = float(oscillation_force)
        self.global_psi += self.last_oscillation_force

        self.last_chaos_kick = self.chaotic_drive()
        self.global_psi += self.last_chaos_kick

        # Rare exogenous shocks to prevent static lock-in
        self.last_shock = 0.0
        if self.rng.random() < self.shock_probability:
            self.last_shock = (self.rng.random() * 2.0 - 1.0) * self.shock_strength
            self.global_psi += self.last_shock

        self.stochastic_update()
        self.broadcast_prompt()

        # 2) Metabolic labor (vectorized)
        metabolic_cost = 0.25 + (self.global_psi * 2.0)
        self.agents_currency -= metabolic_cost

        rand_vals = self.rng.random(self.num_agents)
        success_prob = 0.60 + 0.35 * (self.agents_reputation / 5.0)
        success_mask = rand_vals < success_prob

        reward_val = 1.8 * (1.0 - self.global_psi)
        self.agents_currency[success_mask] += reward_val
        self.agents_reputation[success_mask] = np.minimum(5.0, self.agents_reputation[success_mask] + 0.1)
        self.agents_reputation[~success_mask] = np.maximum(0.1, self.agents_reputation[~success_mask] - 0.05)
        self.agents_currency[~success_mask] -= 0.1

        # 3) Death/rebirth state
        dead_mask = self.agents_currency <= 0
        if np.any(dead_mask):
            num_dead = int(np.sum(dead_mask))
            self.agents_memory[dead_mask] = self.rng.choice([-1.0, 1.0], size=(num_dead, self.dim)).astype(np.float32)
            self.agents_currency[dead_mask] = 5.0
            self.agents_reputation[dead_mask] = 0.5

        # 4) Back-reaction from aggregate work
        work_efficiency = float(np.mean(success_mask))
        u, psi = self.symmetry_measure()
        resonance_now = self.calculate_collective_resonance()
        avg_currency_now = float(np.mean(self.agents_currency))
        avg_rep_now = float(np.mean(self.agents_reputation))
        self.history_resonance.append(float(resonance_now))
        fuzzy = self.fuzzy_control(u=u, psi=psi, resonance=resonance_now, work_efficiency=work_efficiency)

        coherence_backlash = max(0.0, resonance_now - 0.94) * 0.16
        reputation_backlash = max(0.0, avg_rep_now - 3.2) * 0.010
        economy_backlash = max(0.0, avg_currency_now - 85.0) / 3200.0
        self.global_psi += coherence_backlash + reputation_backlash + economy_backlash

        low_psi_lock = max(0.0, 0.05 - self.global_psi) / 0.05
        correction_gain = (0.045 + 0.03 * fuzzy["stabilize_boost"]) * (1.0 - 0.45 * low_psi_lock)
        entropy_return = 0.005 * fuzzy["entropy_relax"] + 0.003 * low_psi_lock
        # Keep bounded but dynamic (avoid hard convergence at near-zero floor)
        self.global_psi = float(np.clip(self.global_psi - (correction_gain * work_efficiency) + entropy_return, 0.0, 0.65))

        # Soft chaotic floor/ceiling to keep non-static oscillation over long runs.
        entropy_floor = 0.02 + 0.40 * abs(self.last_chaos_kick) + 0.18 * abs(self.last_shock)
        if self.global_psi < entropy_floor:
            self.global_psi = entropy_floor
        elif self.global_psi > 0.62:
            self.global_psi = 0.62 - (0.06 + 0.10 * abs(self.last_chaos_kick))

        self.history_psi.append(float(self.global_psi))

        # ── Entropy-death check ─────────────────────────────────────────────
        if not self.collapsed:
            if self.global_psi >= self.collapse_psi_threshold:
                self._collapse_ticks += 1
                if self._collapse_ticks == 1:
                    self.log_data(
                        f"[ ⚠ Entropy critical — Ψ={self.global_psi:.4f} ≥ {self.collapse_psi_threshold} "
                        f"| collapse in {self.collapse_patience} ticks if unresolved ]"
                    )
                elif self._collapse_ticks % 10 == 0:
                    remaining = self.collapse_patience - self._collapse_ticks
                    self.log_data(
                        f"[ ⚠ Entropy sustained — Ψ={self.global_psi:.4f} "
                        f"| {remaining} ticks until heat death ]"
                    )
                if self._collapse_ticks >= self.collapse_patience:
                    self.collapsed = True
                    self.log_data(
                        f"\n[ ✦ UNIVERSE COLLAPSED — "
                        f"Ψ held above {self.collapse_psi_threshold} for {self._collapse_ticks} ticks. "
                        f"Coherence lost at step {self.step}. "
                        f"Agents scattered. The void remains. ✦ ]\n"
                    )
                    if self.on_collapse:
                        self.on_collapse(self.step, self._collapse_ticks)
            else:
                if self._collapse_ticks > 0:
                    self.log_data(
                        f"[ ✓ Entropy receding — Ψ={self.global_psi:.4f} "
                        f"| collapse counter reset (was {self._collapse_ticks}) ]"
                    )
                self._collapse_ticks = 0
        
        if self.step % self.log_every_n_steps == 0:
            u, psi = self.symmetry_measure()
            avg_currency = float(np.mean(self.agents_currency))
            min_currency = float(np.min(self.agents_currency))
            max_currency = float(np.max(self.agents_currency))
            avg_rep = float(np.mean(self.agents_reputation))
            active = int(np.sum(self.agents_currency > 0))
            metabolic_cost = 0.25 + (self.global_psi * 2.0)

            base = (
                f"Step {self.step:4d} | Ψ={psi:.4f} | U≈{u:.4f} | Active={active}/{self.num_agents} | "
                f"WorkEff={work_efficiency:.3f} | Metabolic={metabolic_cost:.3f}"
            )

            if self.verbose_data_log:
                noise = (
                    f"Noise[sto={self.last_stochastic_intensity:.6f}, dyn={self.last_dynamic_noise:.6f}, "
                    f"osc={self.last_oscillation_force:+.5f}, chaos={self.last_chaos_kick:+.5f}, shock={self.last_shock:+.5f}, "
                    f"tor={self.last_toroidal_fraction:.3f}]"
                )
                economy = (
                    f"Economy[cur_avg={avg_currency:.2f}, cur_min={min_currency:.2f}, cur_max={max_currency:.2f}, "
                    f"rep_avg={avg_rep:.3f}]"
                )
                resonance = f"Resonance={resonance_now:.5f}"
                self.log_data(f"{base} | {resonance} | {economy} | {noise}")
            else:
                self.log_data(
                    f"{base} | Currency={avg_currency:.1f} | Resonance={resonance_now:.5f}"
                )

        if self.step % self.mind_read_every_n == 0 and self.concept_matrix is not None:
            results = self.read_population_mind(top_n=5)
            self.log_data(self.format_mind_read(results))
            scenario_state = self.evaluate_population_scenarios()
            self.log_data(self.scenario_vector_string(scenario_state.get("scores", {}), top_n=6))
            if self.on_mind_read:
                self.on_mind_read(results)
    
    def launch(self, steps: int = 400):
        self.log_data("=== AxUniverseSim (Pure Python) Launched ===")
        self.log_data("Equation + Monetary system + Common language + Forced labor active\n")
        self.broadcast_prompt()  # initial prompt
        for _ in range(steps):
            self.run_tick()
            time.sleep(0.015)  # readable speed
        u, psi = self.symmetry_measure()
        self.log_data(f"\n=== Simulation Finished ===\nFinal Ψ = {psi:.5f} | Effective observable U = {u:.4f}")

if __name__ == "__main__":
    sim = AxUniverseSim(dim=1024, num_agents=30)
    sim.run_with_observer()
