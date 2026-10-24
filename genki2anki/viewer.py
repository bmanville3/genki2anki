import csv
import json
from pathlib import Path

from . import runctl

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Genki review</title><style>
:root{--bg:#fafafa;--fg:#222;--card:#fff;--bd:#d8d8d8;--mut:#777;--ac:#3a6ea5;--bad:#c0392b;--ok:#2e8b57}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--card:#1f1f1f;--bd:#333;--mut:#999;--ac:#7fb2e5;--bad:#e57373;--ok:#66bb8a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,"Noto Sans CJK JP",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--bd);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header input[type=search],header select{padding:6px 8px;background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:6px}
button{padding:6px 10px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer}
button.on{background:var(--ac);color:#fff;border-color:var(--ac)}
#count{color:var(--mut);margin-left:auto}
main{max-width:1200px;margin:0 auto;padding:12px 16px;display:grid;gap:12px}
.row{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;display:grid;grid-template-columns:minmax(220px,1fr) minmax(240px,1fr) minmax(240px,1fr);gap:14px}
@media(max-width:900px){.row{grid-template-columns:1fr}}
.front{font-size:34px;line-height:1.2}.sub{color:var(--mut);margin:2px 0}.en{font-size:18px;margin:6px 0}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;border:1px solid var(--bd);margin:2px 3px 2px 0}
.bad{color:var(--bad);border-color:var(--bad)}.ok{color:var(--ok);border-color:var(--ok)}
.shot img,.page img{width:100%;border:1px solid var(--bd);border-radius:6px;background:#fff}
.page img{cursor:zoom-in;max-height:260px;object-fit:cover;object-position:top}.page img.big{max-height:none;cursor:zoom-out}
.lbl{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:6px 0 2px}
textarea{width:100%;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--bd);border-radius:6px;min-height:36px}
.expl{max-height:140px;overflow:auto;font-size:13px;border-top:1px solid var(--bd);margin-top:6px;padding-top:4px}
.expl td,.expl th{border:1px solid var(--bd);padding:1px 5px}
</style></head><body>
<header>
<button id="tv" class="on">Vocab</button><button id="tg">Grammar</button>
<input type="search" id="q" placeholder="search" size="22">
<select id="f"><option value="all">All rows</option><option value="review">Still needs review</option><option value="nv">Needs verification</option><option value="audio">Audio to check</option><option value="done">Verified</option></select>
<select id="l"></select>
<button id="dl">Download review.csv</button><span id="count"></span>
</header><main id="m"></main>
<script>
const D={vocab:__VOCAB__,grammar:__GRAMMAR__};
let tab="vocab";const $=s=>document.querySelector(s);
const st=JSON.parse(localStorage.getItem("g2a_review")||"{}");
const save=()=>localStorage.setItem("g2a_review",JSON.stringify(st));
const rv=r=>{const s=st[r.id]||{};return{v:s.v!==undefined?s.v:(r.verified||"").toUpperCase()==="TRUE",n:s.n!==undefined?s.n:(r.notes||"")}};
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function fill(){const ls=[...new Set(D[tab].map(r=>r.lesson))].sort((a,b)=>a-b);$("#l").innerHTML='<option value="">All lessons</option>'+ls.map(l=>`<option>${esc(l)}</option>`).join("")}
function pass(r){const q=$("#q").value.toLowerCase(),f=$("#f").value,l=$("#l").value,s=rv(r);
if(l&&String(r.lesson)!==l)return false;
if(f==="review"&&!(r.needs_verification==="TRUE"&&!s.v))return false;
if(f==="nv"&&r.needs_verification!=="TRUE")return false;
if(f==="audio"&&r.needs_audio_check!=="TRUE")return false;
if(f==="done"&&!s.v)return false;
return !q||Object.values(r).some(v=>String(v).toLowerCase().includes(q))}
function tags(r){let h="";(r.verify_reasons||"").split(";").filter(Boolean).forEach(x=>h+=`<span class="tag bad">${esc(x)}</span>`);
if(r.needs_verification!=="TRUE")h+='<span class="tag ok">no flags</span>';
if(r.needs_audio_check==="TRUE")h+='<span class="tag bad">check audio</span>';return h}
function card(r){const s=rv(r),v=tab==="vocab";
const left=v?`<div class="front">${esc(r.front)}</div><div class="sub">${esc(r.front_script)} · L${esc(r.lesson)} · ${esc(r.category)}</div>
<div class="en">${esc(r.english)}</div><div class="sub">kanji: ${esc(r.kanji)||"—"} (${esc(r.kanji_source)||"—"})</div>
<div class="sub">hiragana: ${esc(r.hiragana)} · katakana: ${esc(r.katakana)}</div><div class="sub">furigana: ${esc(r.furigana)||"—"}</div>
<div class="sub">romaji: ${esc(r.romaji)} · reading: ${esc(r.reading_source)}</div>
<div class="sub">jisho: ${esc(r.jisho_status)} ${esc(r.jisho_word)} ${esc(r.jisho_reading)} ${esc(r.jlpt)}</div><div class="sub">${esc(r.jisho_english)}</div>`
:`<div class="front" style="font-size:26px">${esc(r.sentence)}</div><div class="sub">${esc(r.hiragana)}</div><div class="sub">${esc(r.furigana)}</div>
<div class="en">${esc(r.english)}</div><div class="sub">L${esc(r.lesson)} · ${esc(r.grammar_number)} ${esc(r.grammar_title)}</div>
<div class="sub">textbook: ${esc(r.sentence_textbook)} (${esc(r.front_source)})</div><div class="expl">${r.explanation_html||""}</div>`;
const aud=r.audio_file?`<audio controls preload="none" src="${esc(r.audio_file)}"></audio><div class="sub">${esc(r.audio_source)} ${esc(r.audio_confidence)} ${esc(r.audio_sim)}</div>`:"";
const pg=r.page_image?`<div class="lbl">Textbook page ${esc(r.page)}</div><div class="page"><img loading="lazy" src="${esc(r.page_image)}" onclick="this.classList.toggle('big')"></div>`:"";
const shot=v?(r.jisho_screenshot?`<div class="lbl">Jisho</div><div class="shot"><a href="${esc(r.jisho_url)}" target="_blank"><img loading="lazy" src="${esc(r.jisho_screenshot)}"></a></div>`:'<div class="lbl">Jisho</div><div class="sub">no screenshot</div>'):"";
return `<div class="row" data-id="${esc(r.id)}"><div>${left}${aud}<div>${tags(r)}</div></div><div>${pg}</div><div>${shot}
<div class="lbl">Review</div><label><input type="checkbox" class="vf" ${s.v?"checked":""}> verified</label><textarea class="nt" placeholder="notes">${esc(s.n)}</textarea></div></div>`}
function render(){const rows=D[tab].filter(pass);$("#count").textContent=rows.length+" / "+D[tab].length;
$("#m").innerHTML=rows.slice(0,300).map(card).join("")+(rows.length>300?`<div class="sub">showing first 300 - narrow the filter</div>`:"")}
$("#m").addEventListener("change",e=>{const id=e.target.closest(".row").dataset.id;st[id]=st[id]||{};if(e.target.classList.contains("vf"))st[id].v=e.target.checked;save()});
$("#m").addEventListener("input",e=>{if(e.target.classList.contains("nt")){const id=e.target.closest(".row").dataset.id;st[id]=st[id]||{};st[id].n=e.target.value;save()}});
["#q","#f","#l"].forEach(s=>$(s).addEventListener("input",render));
function setTab(t){tab=t;$("#tv").classList.toggle("on",t==="vocab");$("#tg").classList.toggle("on",t==="grammar");fill();render()}
$("#tv").onclick=()=>setTab("vocab");$("#tg").onclick=()=>setTab("grammar");
$("#dl").onclick=()=>{const q=s=>'"'+String(s).replace(/"/g,'""')+'"';let o="id,verified,notes\\n";
for(const t of["vocab","grammar"])for(const r of D[t]){const s=rv(r);if(s.v||s.n)o+=[q(r.id),s.v?"TRUE":"",q(s.n)].join(",")+"\\n"}
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([o],{type:"text/csv"}));a.download="review.csv";a.click()};
setTab("vocab");
</script></body></html>"""


def _read(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def _js(rows):
    return json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")


def run_viewer(out_dir):
    out = Path(out_dir)
    html = PAGE.replace("__VOCAB__", _js(_read(out / "vocab.csv"))).replace("__GRAMMAR__", _js(_read(out / "grammar.csv")))
    runctl.atomic_write_text(out / "viewer.html", html)
    print(f"[viewer] {out / 'viewer.html'}")
    return out / "viewer.html"
