import re
import unicodedata
from typing import Dict, List, Tuple

def normalize_text(text: str) -> str:
    """Normalize text: unicode NFKD, lowercase, whitespace clean."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    # Remove accents for Spanish/German to simplify matching
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    return text

# Negation prefix and suffix patterns across languages
NEGATION_PATTERNS = [
    # English
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\bintact\b", r"\bunremarkable\b",
    r"\bnormal\b", r"\bpreserved\b", r"\bfree of\b", r"\babsence of\b", r"\bdenies\b",
    r"\bno evidence of\b", r"\bno sign of\b", r"\bno definite\b", r"\bno frank\b",
    # Spanish
    r"\bsin\b", r"\bno se observa\b", r"\bno hay\b", r"\bintacto\b", r"\bintacta\b",
    r"\bconservado\b", r"\bconservada\b", r"\bnormal\b", r"\bnormales\b", r"\bausencia de\b",
    r"\blimites normales\b", r"\bdescartar\b",
    # German
    r"\bkein\b", r"\bkeine\b", r"\bkeinerlei\b", r"\bohne\b", r"\bregelrecht\b",
    r"\bregelrechte\b", r"\bregelrechtes\b", r"\bintakt\b", r"\bintakte\b", r"\bausschluss\b",
    # Bulgarian / Cyrillic
    r"\bняма\b", r"\bбез\b", r"\bинтактна\b", r"\bинтактен\b", r"\bзапазена\b", r"\bнормален\b"
]

class MultilingualKneeReportExtractor:
    """
    Multilingual clinical entity extractor for 12 knee abnormalities:
    ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
    PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        # 1. ACL patterns
        self.acl_terms = [
            r"acl", r"anterior cruciate ligament", r"lca", r"ligamento cruzado anterior",
            r"vkb", r"vorderes kreuzband", r"пкв", r"предна кр(?:ъ|у)стна вр(?:ъ|у)зка"
        ]
        self.acl_tear_terms = [
            r"tear", r"rupture", r"torn", r"disruption", r"avulsion",
            r"rotura", r"desgarro", r"ruptura",
            r"ruptur", r"riss", r"anriss", r"fissur",
            r"скъсване", r"руптура", r"разкъсване"
        ]
        self.acl_negations = [
            r"intact", r"normal", r"preserved", r"unremarkable", r"continuous",
            r"intacto", r"intacta", r"conservado", r"normales",
            r"intakt", r"regelrecht", r"durchgangig",
            r"интактна", r"интактен", r"запазена", r"б\.о\."
        ]

        # 2. MCL patterns
        self.mcl_terms = [
            r"mcl", r"medial collateral ligament", r"tibial collateral ligament",
            r"lcm", r"ligamento colateral medial", r"ligamento colateral interno",
            r"innenband", r"mediales seitenband", r"мкл", r"медиална колатерална"
        ]
        self.mcl_tear_terms = [
            r"tear", r"rupture", r"torn", r"grade 2", r"grade 3", r"high grade",
            r"rotura", r"desgarro", r"grado 2", r"grado 3",
            r"ruptur", r"riss", r"hohergradig",
            r"скъсване", r"руптура", r"разкъсване"
        ]

        # 3. Medial Meniscus patterns
        self.mm_terms = [
            r"medial meniscus", r"posterior horn of (?:the )?medial meniscus",
            r"menisco medial", r"menisco interno", r"cuerno posterior del menisco interno",
            r"innenmeniskus", r"hinterhorn des innenmeniskus",
            r"медиален менискус"
        ]

        # 4. Lateral Meniscus patterns
        self.lm_terms = [
            r"lateral meniscus", r"posterior horn of (?:the )?lateral meniscus",
            r"menisco lateral", r"menisco externo",
            r"aussenmeniskus", r"hinterhorn des aussenmeniskus",
            r"латерален менискус"
        ]

        self.meniscus_tear_terms = [
            r"tear", r"torn", r"radial tear", r"horizontal tear", r"complex tear",
            r"root tear", r"bucket handle", r"flap", r"maceration", r"substance tear",
            r"rotura", r"desgarro", r"fisura", r"amputacion", r"asa de cubo",
            r"riss", r"ruptur", r"meniskusriss", r"radiaerer riss", r"korbhenkelriss",
            r"скъсване", r"руптура", r"дегенеративна лезия"
        ]

        # 5. Medial OA
        self.moa_terms = [
            r"medial.*?(?:osteoarthritis|oa|arthrosis|chondropathy|chondromalacia|joint space narrowing|cartilage thinning|fissuring)",
            r"artrosis.*?medial", r"artrosis femorotibial medial", r"condropatia.*?medial", r"gonartrosis medial",
            r"medial.*?gonarthrose", r"mediale gonarthrose", r"medial betont.*?arthrose", r"chondropathie femorotibial medial",
            r"медиална.*?(?:гонартроза|остеоартроза)"
        ]

        # 6. Lateral OA
        self.loa_terms = [
            r"lateral.*?(?:osteoarthritis|oa|arthrosis|chondropathy|chondromalacia|joint space narrowing|cartilage thinning|fissuring)",
            r"artrosis.*?lateral", r"artrosis femorotibial lateral", r"condropatia.*?lateral", r"gonartrosis lateral",
            r"lateral.*?gonarthrose", r"laterale gonarthrose", r"lateral betont.*?arthrose", r"chondropathie femorotibial lateral",
            r"латерална.*?(?:гонартроза|остеоартроза)"
        ]

        # 7. PF OA (Patellofemoral)
        self.pfoa_terms = [
            r"patellofemoral.*?(?:osteoarthritis|oa|arthrosis|chondropathy|chondromalacia|cartilage|fissuring)",
            r"trochlea.*?(?:chondropathy|fissuring|thinning)",
            r"patellar facet.*?(?:cartilage|fissuring|thinning)",
            r"artrosis.*?(?:patelofemoral|femororrotuliana)", r"condromalacia rotuliana", r"condropatia.*?(?:rotuliana|patelar)",
            r"retropatellar.*?(?:arthrose|chondropathie|knorpelschaden)",
            r"пателофеморална.*?(?:артроза|остеоартроза)"
        ]

        # 8. Effusion
        self.effusion_terms = [
            r"effusion", r"joint effusion", r"fluid", r"suprapatellar effusion", r"hemarthrosis",
            r"derrame", r"derrame articular", r"liquido intraarticular", r"hemartros",
            r"gelenkerguss", r"erguss", r"flussigkeit",
            r"ставен излив", r"излив", r"течност"
        ]

        # 9. Synovitis
        self.synovitis_terms = [
            r"synovitis", r"synovial thickening", r"synovial hypertrophy", r"synovial proliferation",
            r"sinovitis", r"engrosamiento sinovial", r"hipertrofia sinovial",
            r"synovitis", r"synovialitis", r"synoviale hyperplasie",
            r"синовит", r"синовиално задебеляване"
        ]

        # 10. Baker's Cyst
        self.baker_terms = [
            r"baker'?s cyst", r"popliteal cyst", r"gastrocnemius-semimembranosus bursa",
            r"quiste de baker", r"quiste popliteo",
            r"baker-zyste", r"bakerzyste", r"poplitealzyste",
            r"бейкерова киста", r"поплитеална киста"
        ]

        # 11. Contusion (Bone Bruise / Edema)
        self.contusion_terms = [
            r"contusion", r"bone bruise", r"bone contusion", r"marrow edema", r"trabecular microfracture",
            r"contusion osea", r"edema oseo", r"edema de medula osea",
            r"bone bruise", r"knochenkontusion", r"knochenmarkodem",
            r"костна контузия", r"костен едем", r"костномозъчен едем"
        ]

        # 12. Fracture
        self.fracture_terms = [
            r"fracture", r"insufficiency fracture", r"tibial plateau fracture", r"avulsion fracture",
            r"fractura", r"fisura osea",
            r"fraktur", r"knochenfraktur", r"infraktionslinie",
            r"фрактура", r"пукнатина"
        ]

    def _split_into_sentences_or_clauses(self, text: str) -> List[str]:
        # Split text by periods, colons, newlines, semicolons, bullets
        raw_parts = re.split(r"[\n\r;.]+|\s*[-*•>]\s*", text)
        return [p.strip() for p in raw_parts if len(p.strip()) > 2]

    def _has_negation(self, window: str) -> bool:
        for neg in NEGATION_PATTERNS:
            if re.search(neg, window, re.IGNORECASE):
                return True
        return False

    def extract_from_report(self, raw_report: str) -> Dict[str, float]:
        """
        Extract confidence probabilities [0.0, 1.0] for the 12 knee abnormalities.
        Returns a dictionary mapping label name to float score.
        """
        norm_text = normalize_text(raw_report)
        clauses = self._split_into_sentences_or_clauses(norm_text)

        # Baseline default probabilities (calibrated close to competition prior)
        preds = {
            "ACL": 0.05,
            "MCL": 0.05,
            "Medial Meniscus": 0.10,
            "Lateral Meniscus": 0.10,
            "Medial OA": 0.10,
            "Lateral OA": 0.05,
            "PF OA": 0.10,
            "Effusion": 0.15,
            "Synovitis": 0.05,
            "Baker's": 0.05,
            "Contusion": 0.05,
            "Fracture": 0.05
        }

        # 1. ACL
        for clause in clauses:
            for term in self.acl_terms:
                if re.search(term, clause):
                    # Check if negated/intact
                    is_intact = any(re.search(neg, clause) for neg in self.acl_negations)
                    is_negated = self._has_negation(clause)
                    has_tear = any(re.search(tear, clause) for tear in self.acl_tear_terms)

                    if (is_intact or is_negated) and not has_tear:
                        preds["ACL"] = min(preds["ACL"], 0.02)
                    elif has_tear and not is_negated:
                        preds["ACL"] = 0.98

        # 2. MCL
        for clause in clauses:
            for term in self.mcl_terms:
                if re.search(term, clause):
                    is_negated = self._has_negation(clause)
                    has_tear = any(re.search(tear, clause) for tear in self.mcl_tear_terms)
                    # Note: mild / grade 1 sprains without tear are negative in RSNA ground truth
                    is_grade1 = bool(re.search(r"grade 1|grado 1|mild sprain|low[- ]grade|distension", clause))
                    if is_negated or (is_grade1 and not has_tear):
                        preds["MCL"] = min(preds["MCL"], 0.02)
                    elif has_tear and not is_negated:
                        preds["MCL"] = 0.95

        # 3. Medial Meniscus
        for clause in clauses:
            for term in self.mm_terms:
                if re.search(term, clause):
                    is_negated = self._has_negation(clause)
                    has_tear = any(re.search(tear, clause) for tear in self.meniscus_tear_terms)
                    # "no tear" or "sin rotura" or "intact"
                    if is_negated and not has_tear:
                        preds["Medial Meniscus"] = min(preds["Medial Meniscus"], 0.02)
                    elif has_tear and not is_negated:
                        preds["Medial Meniscus"] = 0.98

        # 4. Lateral Meniscus
        for clause in clauses:
            for term in self.lm_terms:
                if re.search(term, clause):
                    is_negated = self._has_negation(clause)
                    has_tear = any(re.search(tear, clause) for tear in self.meniscus_tear_terms)
                    if is_negated and not has_tear:
                        preds["Lateral Meniscus"] = min(preds["Lateral Meniscus"], 0.02)
                    elif has_tear and not is_negated:
                        preds["Lateral Meniscus"] = 0.98

        # Check combined menisci statements (e.g. "no evidence of tear noted in either medial or lateral meniscus")
        for clause in clauses:
            if re.search(r"either medial or lateral meniscus|ambos meniscos|beide menisken", clause):
                if self._has_negation(clause):
                    preds["Medial Meniscus"] = min(preds["Medial Meniscus"], 0.02)
                    preds["Lateral Meniscus"] = min(preds["Lateral Meniscus"], 0.02)

        # 5. Medial OA
        for clause in clauses:
            for term in self.moa_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Medial OA"] = min(preds["Medial OA"], 0.02)
                    else:
                        preds["Medial OA"] = 0.95

        # 6. Lateral OA
        for clause in clauses:
            for term in self.loa_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Lateral OA"] = min(preds["Lateral OA"], 0.02)
                    else:
                        preds["Lateral OA"] = 0.95

        # 7. PF OA
        for clause in clauses:
            for term in self.pfoa_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["PF OA"] = min(preds["PF OA"], 0.02)
                    else:
                        preds["PF OA"] = 0.95

        # Tricompartmental OA affects all three
        for clause in clauses:
            if re.search(r"tricompartmental|tricompartimental|tri-compartmental|panartrosis", clause):
                if not self._has_negation(clause):
                    preds["Medial OA"] = max(preds["Medial OA"], 0.92)
                    preds["Lateral OA"] = max(preds["Lateral OA"], 0.92)
                    preds["PF OA"] = max(preds["PF OA"], 0.92)

        # 8. Effusion
        for clause in clauses:
            for term in self.effusion_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Effusion"] = min(preds["Effusion"], 0.05)
                    else:
                        preds["Effusion"] = 0.95

        # 9. Synovitis
        for clause in clauses:
            for term in self.synovitis_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Synovitis"] = min(preds["Synovitis"], 0.02)
                    else:
                        preds["Synovitis"] = 0.95

        # 10. Baker's Cyst
        for clause in clauses:
            for term in self.baker_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Baker's"] = min(preds["Baker's"], 0.02)
                    else:
                        preds["Baker's"] = 0.95

        # 11. Contusion
        for clause in clauses:
            for term in self.contusion_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Contusion"] = min(preds["Contusion"], 0.02)
                    else:
                        preds["Contusion"] = 0.95

        # 12. Fracture
        for clause in clauses:
            for term in self.fracture_terms:
                if re.search(term, clause):
                    if self._has_negation(clause):
                        preds["Fracture"] = min(preds["Fracture"], 0.02)
                    else:
                        preds["Fracture"] = 0.95

        return preds
