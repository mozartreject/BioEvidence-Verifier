/* Guard native form navigation even when the main application fails to load. */
(() => {
 const form = document.getElementById('claim-form');
 const button = document.getElementById('submit');
 const error = document.getElementById('error');
 window.biomedicalReady = false;
 const failed = () => {
  if (window.biomedicalReady) return;
  button.disabled = true;
  error.hidden = false;
  error.textContent = 'The interface could not finish loading. Your claim is still here. Refresh this page to load the latest version.';
 };
 form.addEventListener('submit', event => {
  event.preventDefault();
  if (!window.biomedicalReady) failed();
 }, true);
 window.addEventListener('error', failed);
 window.addEventListener('load', () => { if (!window.biomedicalReady) failed(); });
})();
