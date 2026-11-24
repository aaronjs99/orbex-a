// Shared by Studio and standalone generated theory pages. No page navigation.
const installations = new WeakMap();
const MESSAGE = 'orbexa.figure.open';
const imageExtension = /\.(?:png|jpe?g|gif|webp|avif|svg)$/i;

export function localImageURL(value, base) {
  try {
    const url = new URL(value, base), origin = new URL(base).origin;
    return ['http:', 'https:', 'file:'].includes(url.protocol) && url.origin === origin && imageExtension.test(url.pathname) ? url.href : null;
  } catch { return null; }
}

const style = `
.figure-viewer{box-sizing:border-box;width:min(96vw,1800px);max-width:96vw;max-height:94dvh;margin:auto;padding:0;border:1px solid #40576b;border-radius:10px;background:#0b1119;color:#e5edf5;box-shadow:0 24px 100px #000a;font:14px/1.5 "Segoe UI",Arial,sans-serif;overflow:auto}
.figure-viewer:not([open]){display:none}.figure-viewer::backdrop{background:#02070edb;backdrop-filter:blur(5px)}
.figure-viewer .figure-viewer-header{display:flex;align-items:center;gap:16px;padding:12px 16px;border-bottom:1px solid #263342}.figure-viewer .figure-viewer-title{flex:1;margin:0;font-size:14px;line-height:1.5;font-weight:500;overflow-wrap:anywhere}
.figure-viewer .figure-viewer-close{flex-shrink:0;border:1px solid #40576b;border-radius:5px;min-width:36px;min-height:36px;background:#142130;color:#e5edf5;font-size:23px;cursor:pointer}.figure-viewer .figure-viewer-close:hover{border-color:#5bd4bd}.figure-viewer .figure-viewer-close:focus-visible{outline:2px solid #5bd4bd;outline-offset:3px}
.figure-viewer .figure-viewer-image{display:block;width:100%;height:auto;max-height:calc(94dvh - 76px);object-fit:contain;margin:0;background:#0b1119}.figure-viewer .figure-viewer-error{margin:20px;color:#f48f9b}.figure-viewer .figure-viewer-error[hidden]{display:none}
@media(prefers-reduced-motion:no-preference){.figure-viewer[open]{animation:figure-arrive .13s ease-out}@keyframes figure-arrive{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}}
@media(max-width:600px){.figure-viewer{width:98vw;max-width:98vw;border-radius:7px}.figure-viewer .figure-viewer-header{padding:9px 11px}}
`;

export function installImageViewer(root = document, { onOpen, onClose } = {}) {
  const doc = root.nodeType === 9 ? root : root.ownerDocument, win = doc.defaultView;
  if (installations.has(doc)) { const installed = installations.get(doc); installed.attach(root); return installed; }
  let dialog, image, title, error, closeButton, trigger, lockedOverflow, opened = false;
  const roots = new Set();
  const restoreFocus = value => { if (value?.isConnected) value.focus?.({ preventScroll: true }); };
  function cleanup(restore = true) {
    if (!opened) return;
    opened = false;
    doc.documentElement.style.overflow = lockedOverflow;
    const previous = trigger; trigger = null;
    if (restore) restoreFocus(previous);
    onClose?.();
  }
  function close({ restoreFocus: restore = true } = {}) {
    if (!opened) return;
    cleanup(restore);
    if (dialog.open) dialog.close();
  }
  function build() {
    if (dialog) return;
    const css = doc.createElement('style'); css.dataset.orbexaImageViewer = ''; css.textContent = style; doc.head.append(css);
    dialog = doc.createElement('dialog'); dialog.className = 'figure-viewer';
    dialog.setAttribute('aria-label', 'Enlarged figure'); dialog.setAttribute('aria-modal', 'true');
    const header = doc.createElement('div'); header.className = 'figure-viewer-header';
    title = doc.createElement('h2'); title.className = 'figure-viewer-title';
    closeButton = doc.createElement('button'); closeButton.type = 'button'; closeButton.className = 'figure-viewer-close';
    closeButton.textContent = '×'; closeButton.setAttribute('aria-label', 'Close enlarged figure');
    image = doc.createElement('img'); image.className = 'figure-viewer-image';
    error = doc.createElement('p'); error.className = 'figure-viewer-error'; error.hidden = true; error.textContent = 'This figure could not be loaded.';
    image.addEventListener('error', () => { error.hidden = false; });
    image.addEventListener('load', () => { error.hidden = true; });
    header.append(title, closeButton); dialog.append(header, image, error); doc.body.append(dialog);
    closeButton.addEventListener('click', () => close());
    dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
    dialog.addEventListener('close', () => cleanup());
    dialog.addEventListener('click', event => {
      const box = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) close();
    });
  }
  function open(source, { alt = 'Figure', trigger: opener = doc.activeElement } = {}) {
    const href = localImageURL(source, doc.baseURI);
    if (!href) return false;
    // A same-origin Studio iframe uses the parent's full viewport. Standalone
    // theory pages keep their own accessible viewer.
    try {
      if (win.parent !== win && win.parent.document.documentElement.dataset.orbexaImageViewer === 'ready') {
        win.parent.postMessage({ type: MESSAGE, src: href, alt: String(alt).slice(0, 500) }, win.location.origin);
        return true;
      }
    } catch { /* Cross-origin embedding keeps the dialog in this document. */ }
    build();
    if (!opened) { trigger = opener; lockedOverflow = doc.documentElement.style.overflow; }
    title.textContent = String(alt || 'Figure').slice(0, 500); image.alt = title.textContent; error.hidden = true; image.src = href;
    opened = true; doc.documentElement.style.overflow = 'hidden';
    if (!dialog.open) dialog.showModal();
    closeButton.focus({ preventScroll: true }); onOpen?.();
    return true;
  }
  function clicked(event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
    const link = event.target.closest?.('a[href]');
    const picture = link?.querySelector('img') || event.target.closest?.('img[data-orbexa-figure]');
    if ((!link && !picture) || link?.hasAttribute('download') || !event.currentTarget.contains(link || picture)) return;
    const href = localImageURL(link ? link.href : picture.currentSrc || picture.src, doc.baseURI);
    if (!href) return;
    if (open(href, { alt: link?.dataset.figureTitle || picture?.alt || link?.getAttribute('aria-label') || link?.textContent.trim() || 'Figure', trigger: link || picture })) event.preventDefault();
  }
  function message(event) {
    if (event.origin !== win.location.origin || event.data?.type !== MESSAGE) return;
    const frame = [...doc.querySelectorAll('iframe')].find(value => value.contentWindow === event.source);
    if (!frame) return;
    let opener = frame;
    try { opener = frame.contentDocument?.activeElement || frame; } catch { /* Keep the accessible iframe focus target. */ }
    open(event.data.src, { alt: event.data.alt, trigger: opener });
  }
  function keydown(event) {
    if (!opened) {
      const picture = event.target.closest?.('img[data-orbexa-figure]');
      if (picture && ['Enter', ' '].includes(event.key) && open(picture.currentSrc || picture.src, { alt: picture.alt, trigger: picture })) event.preventDefault();
      return;
    }
    if (event.key === 'Escape') { event.preventDefault(); close(); }
    if (event.key === 'Tab') { event.preventDefault(); closeButton.focus(); }
  }
  function enhance(target) {
    for (const picture of target.querySelectorAll('img')) {
      if (picture.closest('a, button') || picture.getAttribute('aria-hidden') === 'true' || !localImageURL(picture.currentSrc || picture.src, doc.baseURI)) continue;
      picture.setAttribute('data-orbexa-figure', ''); picture.setAttribute('role', 'button'); picture.tabIndex = 0;
      picture.setAttribute('aria-haspopup', 'dialog'); picture.setAttribute('aria-label', `Enlarge ${picture.alt || 'figure'}`); picture.style.cursor = 'zoom-in';
    }
  }
  const api = { open, close, isOpen: () => opened,
    attach(target) { if (!roots.has(target)) { roots.add(target); enhance(target); target.addEventListener('click', clicked); } },
    destroy() {
      close(); for (const target of roots) target.removeEventListener('click', clicked);
      roots.clear(); win.removeEventListener('message', message); doc.removeEventListener('keydown', keydown, true);
      dialog?.remove(); doc.documentElement.removeAttribute('data-orbexa-image-viewer'); installations.delete(doc);
    } };
  installations.set(doc, api); doc.documentElement.dataset.orbexaImageViewer = 'ready';
  api.attach(root); win.addEventListener('message', message); doc.addEventListener('keydown', keydown, true);
  return api;
}
