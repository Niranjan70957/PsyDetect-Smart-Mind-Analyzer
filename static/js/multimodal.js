(() => {
  const root = document.getElementById('multimodal-app');
  if (!root) return;
  const $ = id => document.getElementById(`mm-${id}`);
  let selected = null, stream = null, recorder = null, timer = null, previewURL = null;
  let pollGeneration = 0, submitting = false;
  const status = text => { $('status').textContent = text; };
  function stopTracks() { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; }
  function choose(file) {
    if (previewURL) URL.revokeObjectURL(previewURL);
    selected = file; $('preview').srcObject = null; $('preview').muted = false;
    if (file) { previewURL = URL.createObjectURL(file); $('preview').src = previewURL; }
    else { $('preview').removeAttribute('src'); $('preview').load(); }
    $('selection').textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB` : 'No recording selected.';
    $('submit').disabled = !file || submitting;
  }
  $('record').onclick = async () => {
    try {
      if (!navigator.mediaDevices || !window.MediaRecorder) throw new Error('Camera recording is unavailable here. Use localhost or upload a video.');
      $('record').disabled = true;
      stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:true});
      const mime = ['video/webm;codecs=vp8,opus','video/webm','video/mp4'].find(t => MediaRecorder.isTypeSupported(t));
      if (!mime) throw new Error('This browser cannot record a supported video format. Upload MP4 or WebM.');
      selected = null; $('submit').disabled = true; $('file').disabled = true;
      $('preview').srcObject = stream; $('preview').muted = true; await $('preview').play();
      const chunks = [];
      recorder = new MediaRecorder(stream, {mimeType:mime,videoBitsPerSecond:1500000});
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        clearTimeout(timer); stopTracks();
        choose(new File(chunks, `recording.${mime.includes('mp4')?'mp4':'webm'}`, {type:mime}));
        $('stop').disabled = true; $('record').disabled = false; $('file').disabled = false;
        status('Recording ready. Preview it, then choose Analyze.');
      };
      recorder.start(500); $('stop').disabled = false;
      status('Recording… speak naturally. Stops automatically after 30 seconds.');
      timer = setTimeout(() => { if (recorder.state === 'recording') recorder.stop(); },30000);
    } catch (error) {
      stopTracks(); $('record').disabled = false; $('file').disabled = false;
      status(error.name === 'NotAllowedError' ? 'Camera or microphone permission denied. Enable access or upload a video.' : error.message);
    }
  };
  $('stop').onclick = () => { if (recorder && recorder.state === 'recording') recorder.stop(); };
  $('reset').onclick = () => {
    if (recorder && recorder.state === 'recording') { recorder.onstop = () => {}; recorder.stop(); }
    clearTimeout(timer); stopTracks(); choose(null); $('file').value = '';
    $('stop').disabled = true; $('record').disabled = false; $('file').disabled = false;
    status('Ready for another recording.');
  };
  $('file').onchange = () => { choose($('file').files[0] || null); };
  function render(result) {
    $('results').replaceChildren();
    function paragraph(text, title=false) {
      const node = document.createElement(title?'h3':'p'); if (title) node.className='h5 mt-3';
      node.textContent=text; $('results').append(node);
    }
    for (const [key,title] of [['speech_emotion','Voice emotion'],['facial_emotion','Facial expression'],['fused_emotion','Combined emotion']]) {
      paragraph(title,true); const value=result[key];
      paragraph(value ? `${value.label} · ${(value.confidence*100).toFixed(1)}% model confidence` : 'Fusion model has not been trained yet.');
      if(value) paragraph(Object.entries(value.probabilities).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' · '));
    }
    paragraph('Combined depression risk',true);
    paragraph(result.depression.status==='unavailable' ? result.depression.reason : `${result.depression.label} · ${(result.depression.probability*100).toFixed(1)}% estimated probability (experimental)`);
    paragraph(`Face visible in ${(result.quality.face_coverage*100).toFixed(0)}% of sampled frames. Analysis: ${result.elapsed_seconds}s.`);
  }
  async function poll(url, generation) {
    while (generation===pollGeneration) {
      const response=await fetch(url); const data=await response.json();
      if(!response.ok) throw new Error(data.error || 'Could not load analysis.');
      if(generation!==pollGeneration) return;
      if(data.status==='complete') { render(data.result); status('Analysis complete. Saved in your voice + face history.'); return; }
      if(data.status==='failed') throw new Error(data.error);
      status(data.status==='queued'?'Queued for local analysis…':'Analyzing voice and face locally. CPU analysis may take a few minutes…');
      await new Promise(resolve=>setTimeout(resolve,2000));
    }
  }
  $('submit').onclick = async () => {
    if(!selected || submitting) return;
    if(selected.size>100*1048576) { status('Choose a video no larger than 100 MB.'); return; }
    submitting=true; $('submit').disabled=true; $('results').replaceChildren();
    try {
      const body=new FormData(); body.append('video_file',selected); status('Uploading recording…');
      const response=await fetch('/api/multimodal',{method:'POST',headers:{'X-CSRF-Token':root.dataset.csrf},body});
      const data=await response.json(); if(!response.ok) throw new Error(data.error || 'Upload failed.');
      await poll(data.status_url,++pollGeneration);
    } catch(error) { status(error.message); }
    finally { submitting=false; $('submit').disabled=!selected; }
  };
  document.querySelectorAll('.mm-history').forEach(button=>button.onclick=()=>{
    poll(`/api/multimodal/${button.dataset.job}`,++pollGeneration).catch(error=>status(error.message));
  });
  window.addEventListener('beforeunload',stopTracks);
})();
