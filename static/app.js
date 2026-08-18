
const root=document.documentElement;
const saved=localStorage.getItem("moviesense-theme");
if(saved) root.dataset.theme=saved;
const theme=document.getElementById("themeToggle");
if(theme) theme.addEventListener("click",()=>{const next=root.dataset.theme==="dark"?"light":"dark";root.dataset.theme=next;localStorage.setItem("moviesense-theme",next);});

const menuButton=document.getElementById("menuButton"), dropdown=document.getElementById("menuDropdown");
if(menuButton&&dropdown){
  menuButton.addEventListener("click",(e)=>{e.stopPropagation(); const open=dropdown.classList.toggle("show"); menuButton.setAttribute("aria-expanded",open);});
  document.addEventListener("click",()=>dropdown.classList.remove("show"));
  dropdown.addEventListener("click",e=>e.stopPropagation());
}
const text=document.getElementById("reviewText"), count=document.getElementById("charCount");
function updateCount(){if(text&&count) count.textContent=`${text.value.length} / 5000`;}
if(text){text.addEventListener("input",updateCount);updateCount();}
document.getElementById("clearButton")?.addEventListener("click",()=>{text.value="";updateCount();text.focus();});
document.querySelectorAll("[data-sample]").forEach(b=>b.addEventListener("click",()=>{text.value=b.dataset.sample;updateCount();text.focus();}));

const mic=document.getElementById("micButton");
if(mic&&("webkitSpeechRecognition" in window || "SpeechRecognition" in window)){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const recognition=new SR(); recognition.continuous=true; recognition.interimResults=true;
  recognition.onstart=()=>{mic.classList.add("recording");mic.querySelector("span").textContent="Listening…";};
  recognition.onend=()=>{mic.classList.remove("recording");mic.querySelector("span").textContent="Voice Review";};
  recognition.onresult=(e)=>{let finalText="";for(let i=e.resultIndex;i<e.results.length;i++){if(e.results[i].isFinal) finalText+=e.results[i][0].transcript+" ";} if(finalText){text.value+=(text.value?" ":"")+finalText.trim();updateCount();}};
  mic.addEventListener("click",()=>{try{recognition.start();}catch(_){recognition.stop();}});
}else if(mic){mic.title="Voice input is supported in Chrome/Edge";mic.addEventListener("click",()=>alert("Voice review is supported in Chrome or Edge. Please allow microphone access."));}

document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener("click",e=>{const id=a.getAttribute("href");const el=document.querySelector(id);if(el){e.preventDefault();el.scrollIntoView({behavior:"smooth",block:"start"});}}));
