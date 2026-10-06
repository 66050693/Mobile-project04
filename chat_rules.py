"""ตอบคำถามในแชทจากข้อมูลในระบบ โดยไม่ต้องใช้ AI

ใช้เมื่อไม่มี GEMINI_API_KEY หรือ Gemini เรียกไม่สำเร็จ (เช่น 503 ผู้ใช้หนาแน่น)
ตอบได้: แพงไหม/คุ้มไหม, รุ่นนี้มีดีอะไร, เทียบรุ่น/อันดับ, ทำไมไม่ติดอันดับ, ถ้าเพิ่มงบ, แนะนำรุ่นไหน
ทุกตัวเลขมาจากข้อมูลและสูตรเดียวกับหน้าผลลัพธ์
"""
from __future__ import annotations

import re

import pandas as pd

from insight_service import budget_upgrade, why_not
from scoring_service import DIMENSIONS, filter_candidates, pros_cons, score, spec_text

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
ORDINAL = {"หนึ่ง": 1, "แรก": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5}


def _baht(x) -> str:
    return "ไม่มีข้อมูล" if x is None or pd.isna(x) else f"฿{x:,.0f}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().translate(THAI_DIGITS)).strip()


def _short(name: str, brand: str) -> str:
    n = str(name)
    return n[len(brand) + 1:] if n.lower().startswith(str(brand).lower() + " ") else n


def find_phones(question: str, phones: pd.DataFrame, limit: int = 3) -> list[str]:
    """หาชื่อรุ่นที่ถูกพูดถึงในคำถาม (เทียบทั้งชื่อเต็มและชื่อไม่มีแบรนด์ เลือกชื่อที่ยาวที่สุดก่อน)"""
    q = _norm(question)
    cands = []
    for name, brand in zip(phones["name"], phones["brand"]):
        short = _norm(_short(name, brand))
        keys = {_norm(name), short}
        words = short.split()
        if len(words) >= 3 and any(ch.isdigit() for ch in words[-2]):  # ชื่อย่อ เช่น "x9 pro" จาก "find x9 pro"
            keys.add(" ".join(words[-2:]))
        for key in keys:
            if len(key) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9+])", q):
                cands.append((len(key), name, key))
    cands.sort(reverse=True)
    found, used = [], []
    for _, name, key in cands:  # ตัดชื่อที่เป็นส่วนหนึ่งของชื่อที่ยาวกว่า เช่น "x9" ใน "x9 ultra"
        if name not in found and not any(key in u for u in used):
            found.append(name)
            used.append(key)
        if len(found) >= limit:
            break
    return found


def find_ranks(question: str) -> list[int]:
    q = _norm(question)
    ranks = [int(n) for n in re.findall(r"อันดับ\s*(?:ที่\s*)?(\d+)", q)]
    ranks += [ORDINAL[w] for w in re.findall(r"อันดับ\s*(?:ที่\s*)?(หนึ่ง|แรก|สอง|สาม|สี่|ห้า)", q)]
    ranks += [int(n) for n in re.findall(r"(?:กับ|และ|vs|,)\s*(\d)\b", q)] if ranks else []
    return list(dict.fromkeys(r for r in ranks if 1 <= r <= 10))


class RuleChat:
    def __init__(self, phones: pd.DataFrame, req, ranked: pd.DataFrame | None = None):
        self.phones, self.req = phones, req
        self.ranked = ranked if ranked is not None else score(filter_candidates(phones, req), req)
        self._wide = None

    def _row(self, name: str) -> pd.Series:
        """แถวที่มีคะแนนรายด้าน: ถ้าอยู่ในกลุ่มที่จัดอันดับใช้แถวนั้น ไม่งั้นให้คะแนนเทียบกับทุกรุ่นในฐานข้อมูล"""
        hit = self.ranked[self.ranked["name"] == name]
        if len(hit):
            return hit.iloc[0]
        if self._wide is None:
            self._wide = score(self.phones, self.req)
        row = self._wide[self._wide["name"] == name].iloc[0].copy()
        row["_scope"] = "all"
        return row

    def _rank_of(self, name: str) -> str:
        idx = self.ranked.index[self.ranked["name"] == name]
        return f"อันดับ {idx[0] + 1} จาก {len(self.ranked)} รุ่นที่ตรงเงื่อนไข" if len(idx) else "ไม่อยู่ในกลุ่มที่ตรงเงื่อนไขตอนนี้"

    # --- คำตอบแต่ละแบบ ---
    def price(self, name: str) -> str:
        r = self._row(name)
        kind = "ราคาไทย" if r.get("price_kind") == "ราคาไทย" else "ราคาประมาณจากราคาอินเดีย"
        lines = [f"**{name}** ราคา {_baht(r['price_thb'])} ({kind})"]
        if pd.notna(r.get("fair_price_thb")):
            lines.append(f"- โมเดล ML ประเมินว่าสเปกนี้ควรอยู่ที่ราว {_baht(r['fair_price_thb'])} → **{r.get('deal_label') or '-'}**")
        if r.get("segment"):
            lines.append(f"- อยู่ในกลุ่ม: {r['segment']}")
        lines.append("- ราคาที่ควรเป็นเทียบจากสเปกตัวเลข ไม่รวมแบรนด์ การอัปเดต หรือคุณภาพจริง ควรดูรีวิวประกอบ")
        return "\n".join(lines)

    def strengths(self, name: str) -> str:
        r = self._row(name)
        pros, cons = pros_cons(r, self.req)
        specs = ", ".join(x for x in (spec_text(r, d) for d in ("performance", "camera", "battery", "charging", "display")) if x)
        lines = [f"**{name}** ({_baht(r['price_thb'])}) {self._rank_of(name)}", f"- สเปกหลัก: {specs or 'ไม่มีข้อมูล'}"]
        if r.get("_scope") == "all":
            lines.append("- (ข้อดี/ข้อควรรู้ด้านล่างเทียบกับทุกรุ่นในฐานข้อมูล เพราะรุ่นนี้ไม่ผ่านเงื่อนไขที่ตั้งไว้)")
        lines.append("- ข้อดี: " + ("; ".join(pros) if pros else "ไม่มีด้านที่เด่นชัดเมื่อเทียบกับรุ่นอื่นในกลุ่ม"))
        if cons:
            lines.append("- ข้อควรรู้: " + "; ".join(cons))
        return "\n".join(lines)

    def compare(self, names: list[str]) -> str:
        rows = [self._row(n) for n in names]
        head = "| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
        body = [f"| ราคา | " + " | ".join(_baht(r["price_thb"]) for r in rows) + " |"]
        if all("score" in r and pd.notna(r.get("score")) for r in rows):
            body.append("| ความเหมาะสม | " + " | ".join(f"{r['score']:.0f}/100" for r in rows) + " |")
        for d in ("performance", "camera", "selfie", "battery", "charging", "display", "storage"):
            body.append(f"| {DIMENSIONS[d][2]} | " + " | ".join(spec_text(r, d) or "ไม่มีข้อมูล" for r in rows) + " |")
        body.append("| ราคาเทียบสเปก | " + " | ".join(r.get("deal_label") or "-" for r in rows) + " |")
        best = max(rows, key=lambda r: r.get("score") if pd.notna(r.get("score")) else -1)
        return head + "\n" + "\n".join(body) + f"\n\nตามเงื่อนไขที่ตั้งไว้ **{best['name']}** เหมาะกว่า"

    def why(self, name: str) -> str:
        wn = why_not(self.phones, self.req, name)
        return f"**{name}**\n" + "\n".join(f"- {x}" for x in wn["reasons"])

    def upgrade(self) -> str:
        ups = budget_upgrade(self.phones, self.req)
        if not ups:
            return "เพิ่มงบอีก 2,000–5,000 บาท ยังไม่ได้รุ่นที่ดีขึ้นอย่างชัดเจน ใช้งบเดิมคุ้มกว่า"
        return "\n".join(f"- เพิ่มอีก ฿{u['extra']:,} ได้ **{u['name']}** ({_baht(u['price'])}) คะแนนสูงขึ้น {u['gain']:.0f}"
                         + (f" เด่นเรื่อง{', '.join(u['better_at'])}" if u["better_at"] else "") for u in ups)

    def top(self, n: int = 3) -> str:
        if self.ranked.empty:
            return "ยังไม่มีรุ่นที่ตรงเงื่อนไข ลองขยายงบหรือลดตัวกรอง"
        lines = [f"รุ่นที่เหมาะที่สุดตามเงื่อนไขตอนนี้ ({self.req.use_case}):"]
        for i, (_, r) in enumerate(self.ranked.head(n).iterrows(), 1):
            lines.append(f"{i}. **{r['name']}** {_baht(r['price_thb'])} ความเหมาะสม {r['score']:.0f}/100")
        return "\n".join(lines)

    def answer(self, question: str) -> str:
        q = _norm(question)
        names = find_phones(question, self.phones)
        ranks = find_ranks(question)
        for k in ranks:
            if k <= len(self.ranked):
                names.append(self.ranked.iloc[k - 1]["name"])
        names = list(dict.fromkeys(names))

        if re.search(r"เพิ่มงบ|งบเพิ่ม|อัปงบ|อัพงบ|ขยายงบ", q):
            return self.upgrade()
        if names and re.search(r"ทำไม.*(ไม่|ไม่ติด)|ไม่ติด", q):
            return self.why(names[0])
        if len(names) >= 2:
            return self.compare(names[:3])
        if names and re.search(r"แพง|ถูก|คุ้ม|ราคา|เท่าไร|เท่าไหร่", q):
            return self.price(names[0])
        if names:
            return self.strengths(names[0])
        if re.search(r"แนะนำ|รุ่นไหน|ตัวไหน|อันไหน|ซื้ออะไร|ดีสุด|ที่สุด", q):
            return self.top()
        return ("ตอนนี้ตอบจากข้อมูลในระบบ (ไม่ได้ใช้ AI) ลองถามแบบนี้ได้:\n"
                "- OPPO Find X9 แพงไหม\n- Galaxy A57 มีดีอะไร\n- อันดับ 1 กับ 2 ต่างกันยังไง\n"
                "- ทำไมไม่ใช่ iPhone 17\n- ถ้าเพิ่มงบได้อะไร\n- แนะนำรุ่นไหนดี")
