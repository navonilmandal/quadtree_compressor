// main.js — AJAX upload and UI handling
document.addEventListener('DOMContentLoaded', ()=> {
  const form = document.getElementById('uploadForm');
  const compressBtn = document.getElementById('compressBtn');
  const spinner = document.getElementById('spinner');
  const status = document.getElementById('status');
  const resultCard = document.getElementById('resultCard');
  const origPreview = document.getElementById('origPreview');
  const reconPreview = document.getElementById('reconPreview');

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    status.innerText = '';
    resultCard.classList.add('hidden');
    spinner.classList.remove('hidden');
    compressBtn.disabled = true;

    const fd = new FormData(form);
    // show local preview of original
    const f = document.getElementById('fileInput').files[0];
    if(f){
      const url = URL.createObjectURL(f);
      origPreview.src = url;
    }

    try {
      // send form via fetch; server returns full HTML by default, but we accept JSON if requested
      const resp = await fetch(form.action, {
        method: 'POST',
        body: fd,
        headers: { 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'application/json' }
      });
      if(!resp.ok) {
        const txt = await resp.text();
        throw new Error('Server error: ' + resp.status + ' ' + txt.substring(0,200));
      }
      const data = await resp.json();

      // populate UI
      reconPreview.src = 'data:image/png;base64,' + data.recon_b64;
      document.getElementById('psnr').innerText = data.psnr;
      document.getElementById('estSize').innerText = data.estimated;
      document.getElementById('nodes').innerText = data.nodes;
      document.getElementById('usedTol').innerText = data.used_tol;
      document.getElementById('msg').innerText = data.msg || '';

      // download links
      document.getElementById('downloadRecon').href = '/download/recon/' + data.recon_name;
      document.getElementById('downloadJson').href = '/download/json/' + data.json_name;

      resultCard.classList.remove('hidden');
      status.innerText = 'Done';
    } catch(err) {
      console.error(err);
      status.innerText = 'Error: ' + err.message;
      alert('Compression failed: ' + err.message);
    } finally {
      spinner.classList.add('hidden');
      compressBtn.disabled = false;
    }
  });
});
