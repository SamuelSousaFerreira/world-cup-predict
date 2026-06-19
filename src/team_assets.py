"""Bandeiras e cores das seleções (para a interface).

- `FLAG`   : nome da seleção (em inglês, como na base) -> código flagcdn.
- `COLORS` : nome da seleção -> (cor primária, cor secundária) em hex.

Helpers de bandeira usam o CDN público flagcdn.com (imagens PNG por código
ISO 3166-1 alpha-2; subdivisões do Reino Unido via "gb-eng", "gb-sct"...).
Seleções sem mapeamento simplesmente não exibem bandeira/cor (fallback).
"""
from __future__ import annotations

# --------------------------- Bandeiras (flagcdn) ---------------------------- #
# Códigos ISO 3166-1 alpha-2 (minúsculos). Nomes seguem a base martj42.
FLAG: dict[str, str] = {
    "Brazil": "br", "Argentina": "ar", "France": "fr", "England": "gb-eng",
    "Spain": "es", "Germany": "de", "Portugal": "pt", "Netherlands": "nl",
    "Italy": "it", "Belgium": "be", "Croatia": "hr", "Uruguay": "uy",
    "Mexico": "mx", "United States": "us", "Colombia": "co", "Morocco": "ma",
    "Japan": "jp", "South Korea": "kr", "North Korea": "kp", "Senegal": "sn",
    "Switzerland": "ch", "Denmark": "dk", "Poland": "pl", "Sweden": "se",
    "Wales": "gb-wls", "Scotland": "gb-sct", "Northern Ireland": "gb-nir",
    "Republic of Ireland": "ie", "Serbia": "rs", "Ghana": "gh", "Nigeria": "ng",
    "Cameroon": "cm", "Ecuador": "ec", "Peru": "pe", "Chile": "cl",
    "Austria": "at", "Ukraine": "ua", "Turkey": "tr", "Norway": "no",
    "Czech Republic": "cz", "Russia": "ru", "Greece": "gr", "Egypt": "eg",
    "Australia": "au", "Canada": "ca", "Saudi Arabia": "sa", "Qatar": "qa",
    "Ivory Coast": "ci", "Tunisia": "tn", "Algeria": "dz", "Paraguay": "py",
    "Costa Rica": "cr", "IR Iran": "ir", "Iran": "ir", "Iceland": "is",
    "Finland": "fi", "Hungary": "hu", "Romania": "ro", "Slovakia": "sk",
    "Slovenia": "si", "Bosnia and Herzegovina": "ba", "North Macedonia": "mk",
    "Albania": "al", "Bulgaria": "bg", "Israel": "il", "Georgia": "ge",
    "Armenia": "am", "Azerbaijan": "az", "Kazakhstan": "kz", "Belarus": "by",
    "Montenegro": "me", "Luxembourg": "lu", "Cyprus": "cy", "Estonia": "ee",
    "Latvia": "lv", "Lithuania": "lt", "Moldova": "md", "Kosovo": "xk",
    "Mali": "ml", "Burkina Faso": "bf", "DR Congo": "cd", "Congo": "cg",
    "Cape Verde": "cv", "Guinea": "gn", "Gabon": "ga", "South Africa": "za",
    "Zambia": "zm", "Uganda": "ug", "Kenya": "ke", "Angola": "ao",
    "Mauritania": "mr", "Benin": "bj", "Madagascar": "mg", "Equatorial Guinea": "gq",
    "Togo": "tg", "Mozambique": "mz", "Zimbabwe": "zw", "Namibia": "na",
    "Tanzania": "tz", "Sudan": "sd", "Libya": "ly", "Ethiopia": "et",
    "Sierra Leone": "sl", "Niger": "ne", "Comoros": "km", "Gambia": "gm",
    "United Arab Emirates": "ae", "Iraq": "iq", "Oman": "om", "Jordan": "jo",
    "Bahrain": "bh", "Kuwait": "kw", "Lebanon": "lb", "Syria": "sy",
    "Palestine": "ps", "Yemen": "ye", "China PR": "cn", "China": "cn",
    "India": "in", "Thailand": "th", "Vietnam": "vn", "Indonesia": "id",
    "Malaysia": "my", "Philippines": "ph", "Uzbekistan": "uz", "Tajikistan": "tj",
    "Turkmenistan": "tm", "Kyrgyzstan": "kg", "Jamaica": "jm", "Panama": "pa",
    "Honduras": "hn", "El Salvador": "sv", "Guatemala": "gt", "Nicaragua": "ni",
    "Haiti": "ht", "Trinidad and Tobago": "tt", "Curaçao": "cw", "Suriname": "sr",
    "Venezuela": "ve", "Bolivia": "bo", "New Zealand": "nz", "Fiji": "fj",
}

# --------------------------- Cores (primária, secundária) ------------------- #
COLORS: dict[str, tuple[str, str]] = {
    "Brazil": ("#FCC500", "#009C3B"), "Argentina": ("#75AADB", "#0E3C7B"),
    "France": ("#21304F", "#ED2939"), "England": ("#CE1124", "#012169"),
    "Spain": ("#C60B1E", "#FFC400"), "Germany": ("#262626", "#DD0000"),
    "Portugal": ("#1A7A3D", "#DA291C"), "Netherlands": ("#F36C21", "#21468B"),
    "Italy": ("#0E68B2", "#1F8A50"), "Belgium": ("#E30613", "#FDDA24"),
    "Croatia": ("#D81E05", "#1568BD"), "Uruguay": ("#56A0D3", "#0E3C7B"),
    "Mexico": ("#0B6E4F", "#CE1126"), "United States": ("#3C3B6E", "#B22234"),
    "Colombia": ("#F4C20D", "#003893"), "Morocco": ("#C1272D", "#006233"),
    "Japan": ("#BC002D", "#1A1A1A"), "South Korea": ("#CD2E3A", "#0047A0"),
    "Senegal": ("#00853F", "#FDEF42"), "Switzerland": ("#D52B1E", "#FFFFFF"),
    "Denmark": ("#C8102E", "#FFFFFF"), "Poland": ("#DC143C", "#1A1A1A"),
    "Sweden": ("#1E6CB4", "#FECC00"), "Wales": ("#C8102E", "#00B140"),
    "Serbia": ("#C6363C", "#0C4076"), "Ghana": ("#1A7A3D", "#FCD116"),
    "Nigeria": ("#008751", "#1A8E4A"), "Cameroon": ("#1A7A4E", "#CE1126"),
    "Ecuador": ("#F4C20D", "#0033A0"), "Peru": ("#D91023", "#1A1A1A"),
    "Chile": ("#D52B1E", "#0039A6"), "Austria": ("#ED2939", "#1A1A1A"),
    "Ukraine": ("#1E6CB4", "#FFD500"), "Turkey": ("#E30A17", "#1A1A1A"),
    "Norway": ("#BA0C2F", "#00205B"), "Czech Republic": ("#11457E", "#D7141A"),
    "Russia": ("#0039A6", "#D52B1E"), "Greece": ("#0D5EAF", "#1A6FCF"),
    "Egypt": ("#CE1126", "#1A1A1A"), "Australia": ("#1A8E4A", "#FFCD00"),
    "Canada": ("#D80621", "#1A1A1A"), "Saudi Arabia": ("#006C35", "#1A8E4A"),
    "Qatar": ("#8A1538", "#1A1A1A"), "Ivory Coast": ("#FF8200", "#009E60"),
    "Tunisia": ("#E70013", "#1A1A1A"), "Algeria": ("#006233", "#1A8E4A"),
    "Paraguay": ("#D52B1E", "#0038A8"), "Costa Rica": ("#1A4F9E", "#CE1126"),
    "IR Iran": ("#239F40", "#DA0000"), "Iran": ("#239F40", "#DA0000"),
    "Iceland": ("#02529C", "#DC1E35"), "Finland": ("#1E6CB4", "#FFFFFF"),
    "Hungary": ("#C8102E", "#1A7A3D"), "Romania": ("#1A4F9E", "#FCD116"),
    "Slovakia": ("#1E5AA8", "#EE1C25"), "Slovenia": ("#1E5AA8", "#EE1C25"),
    "South Africa": ("#1A8E4A", "#FFB915"), "Jamaica": ("#1A8E4A", "#FED100"),
    "Venezuela": ("#7B1113", "#FCD116"), "Bolivia": ("#1A8E4A", "#D52B1E"),
    "United Arab Emirates": ("#00732F", "#D7141A"), "Iraq": ("#1A8E4A", "#1A1A1A"),
    "Mali": ("#1A8E4A", "#FCD116"), "Burkina Faso": ("#1A8E4A", "#EF2B2D"),
    "DR Congo": ("#1A7CC4", "#F7D618"), "Cape Verde": ("#1A50A8", "#1A1A1A"),
    "China PR": ("#DE2910", "#FFDE00"), "China": ("#DE2910", "#FFDE00"),
    "New Zealand": ("#1A1A1A", "#CE1124"),
}

DEFAULT_HOME = "#2563eb"   # azul
DEFAULT_AWAY = "#dc2626"   # vermelho
DRAW_COLOR = "#94a3b8"     # cinza

_ALLOWED_HEIGHTS = (20, 24, 40, 60, 80, 120, 240)


def flag_code(team: str) -> str | None:
    return FLAG.get(team)


def flag_url(team: str, height: int = 80) -> str | None:
    """URL PNG da bandeira no flagcdn (escolhe a altura suportada mais próxima)."""
    code = FLAG.get(team)
    if not code:
        return None
    h = min(_ALLOWED_HEIGHTS, key=lambda x: abs(x - height))
    return f"https://flagcdn.com/h{h}/{code}.png"


def flag_img_tag(team: str, height: int = 26) -> str:
    """Tag <img> da bandeira (retina), ou string vazia se não houver código."""
    url = flag_url(team, height * 2)
    if not url:
        return ""
    return (
        f"<img src='{url}' height='{height}' "
        "style='vertical-align:middle;border-radius:3px;"
        "box-shadow:0 0 2px rgba(0,0,0,.35);margin-right:8px'>"
    )


def team_color(team: str, default: str = DEFAULT_HOME) -> str:
    c = COLORS.get(team)
    return c[0] if c else default


def team_secondary(team: str) -> str | None:
    c = COLORS.get(team)
    return c[1] if c else None


# ------------------------------ Utilidades de cor --------------------------- #
def _rgb(hexc: str) -> tuple[int, int, int]:
    h = hexc.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _dist(c1: str, c2: str) -> float:
    return sum((a - b) ** 2 for a, b in zip(_rgb(c1), _rgb(c2))) ** 0.5


def lighten(hexc: str, amount: float = 0.35) -> str:
    """Clareia uma cor misturando-a com branco (amount em [0,1])."""
    r, g, b = _rgb(hexc)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def pair_colors(home: str, away: str) -> tuple[str, str]:
    """Cores das duas seleções garantindo contraste entre elas."""
    hc = team_color(home, DEFAULT_HOME)
    ac = team_color(away, DEFAULT_AWAY)
    if _dist(hc, ac) >= 90:
        return hc, ac
    # Muito parecidas: tenta a secundária do visitante, depois do mandante.
    a2 = team_secondary(away)
    if a2 and _dist(hc, a2) >= 90:
        return hc, a2
    h2 = team_secondary(home)
    if h2 and _dist(h2, ac) >= 90:
        return h2, ac
    return DEFAULT_HOME, DEFAULT_AWAY
