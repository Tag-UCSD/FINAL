import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
    X, ChevronLeft, ChevronRight, ShoppingCart, Check,
    Loader2, AlertCircle, ChevronDown, ChevronUp,
} from 'lucide-react';
import { ApiClient } from '@shared';

const api = new ApiClient('/api/v1/explorer');

// ─── Debug mode registry ──────────────────────────────────────────────────────

const DEBUG_MODES = [
    { key: 'none',         label: 'Original',     shortcut: '1', hasEdgeSliders: false, hasOverlaySlider: false, hasSegSlider: false },
    { key: 'edges',        label: 'Edges',         shortcut: '2', hasEdgeSliders: true,  hasOverlaySlider: false, hasSegSlider: false },
    { key: 'overlay',      label: 'Overlay',       shortcut: '3', hasEdgeSliders: true,  hasOverlaySlider: true,  hasSegSlider: false },
    { key: 'depth',        label: 'Depth',         shortcut: '4', hasEdgeSliders: false, hasOverlaySlider: false, hasSegSlider: false },
    { key: 'complexity',   label: 'Complexity',    shortcut: '5', hasEdgeSliders: true,  hasOverlaySlider: false, hasSegSlider: false },
    { key: 'segmentation', label: 'Segmentation',  shortcut: '6', hasEdgeSliders: false, hasOverlaySlider: false, hasSegSlider: true  },
    { key: 'room',         label: 'Room Type',     shortcut: '7', hasEdgeSliders: false, hasOverlaySlider: false, hasSegSlider: false },
    { key: 'materials',    label: 'Materials',     shortcut: '8', hasEdgeSliders: false, hasOverlaySlider: false, hasSegSlider: false },
];

function buildDebugSrc(img, mode, edgeLow, edgeHigh, segConf) {
    if (!img) return '';
    const id = img.id;
    switch (mode) {
        case 'edges':        return `/api/v1/debug/images/${id}/edges?t1=${edgeLow}&t2=${edgeHigh}`;
        case 'depth':        return `/api/v1/debug/images/${id}/depth`;
        case 'complexity':   return `/api/v1/debug/images/${id}/complexity?t1=${edgeLow}&t2=${edgeHigh}`;
        case 'segmentation': return `/api/v1/debug/images/${id}/segmentation?conf=${segConf}`;
        case 'room':         return `/api/v1/debug/images/${id}/room`;
        case 'materials':    return `/api/v1/debug/images/${id}/materials`;
        default:             return img.url;
    }
}

// ─── Attribute group component ────────────────────────────────────────────────

function AttributeGroup({ prefix, attrs, defaultOpen }) {
    const [open, setOpen] = useState(defaultOpen);

    return (
        <div className="border border-gray-100 rounded overflow-hidden mb-1.5">
            <button
                className="w-full flex items-center justify-between px-3 py-1.5 bg-gray-50 hover:bg-gray-100 transition-colors"
                onClick={() => setOpen(v => !v)}
            >
                <span className="text-[10px] font-bold uppercase tracking-widest text-gray-500">
                    {prefix}
                </span>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400">{attrs.length}</span>
                    {open ? <ChevronUp size={12} className="text-gray-400" /> : <ChevronDown size={12} className="text-gray-400" />}
                </div>
            </button>

            {open && (
                <div className="divide-y divide-gray-50">
                    {attrs.map(attr => (
                        <div
                            key={attr.key}
                            className="flex items-center gap-2 px-3 py-1 hover:bg-blue-50 group"
                            title={attr.key}
                        >
                            <span className="text-xs text-gray-600 truncate flex-1 font-mono">
                                {attr.name !== attr.key
                                    ? attr.name
                                    : attr.key.split('.').slice(1).join('.')}
                            </span>
                            <div className="flex items-center gap-2 flex-shrink-0">
                                {/* Mini bar */}
                                <div className="w-14 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-blue-500 rounded-full"
                                        style={{ width: `${Math.min(100, Math.max(0, attr.value * 100))}%` }}
                                    />
                                </div>
                                <span className="text-xs font-mono text-gray-800 w-9 text-right tabular-nums">
                                    {attr.value.toFixed(2)}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

// ─── Tag badge with provenance tooltip ───────────────────────────────────────

function TagBadge({ tag }) {
    const ref = useRef(null);
    const [tipPos, setTipPos] = useState(null);

    const isPreloaded = tag.source === 'preloaded';

    const showTip = () => {
        if (!ref.current) return;
        const r = ref.current.getBoundingClientRect();
        // Position above the tag; clamp to viewport left edge
        setTipPos({ top: r.top - 6, left: Math.max(8, r.left) });
    };

    return (
        <span
            ref={ref}
            onMouseEnter={showTip}
            onMouseLeave={() => setTipPos(null)}
            className={`inline-block text-[9px] uppercase font-bold px-2 py-0.5 rounded cursor-default select-none transition-colors ${
                isPreloaded
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-indigo-900 text-indigo-300 ring-1 ring-inset ring-indigo-700 hover:bg-indigo-800'
            }`}
        >
            {tag.label}

            {/* Fixed-position tooltip — escapes any overflow clipping */}
            {tipPos && (
                <div
                    style={{
                        position: 'fixed',
                        top: tipPos.top,
                        left: tipPos.left,
                        transform: 'translateY(-100%)',
                        zIndex: 9999,
                    }}
                    className="pointer-events-none w-max max-w-[220px] bg-gray-900 border border-gray-600 rounded shadow-2xl px-2.5 py-2"
                >
                    <div className="flex items-center gap-1.5 mb-1">
                        <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${isPreloaded ? 'bg-gray-400' : 'bg-indigo-400'}`} />
                        <span className="text-[10px] font-semibold text-gray-100">{tag.source_label}</span>
                    </div>
                    {tag.confidence != null && (
                        <div className="flex items-center gap-1 mt-0.5">
                            <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-indigo-500 rounded-full"
                                    style={{ width: `${Math.round(tag.confidence * 100)}%` }}
                                />
                            </div>
                            <span className="text-[10px] font-mono text-gray-300 flex-shrink-0">
                                {Math.round(tag.confidence * 100)}%
                            </span>
                        </div>
                    )}
                    {tag.attribute_key && (
                        <div className="text-[9px] font-mono text-indigo-400 mt-1 truncate">
                            {tag.attribute_key}
                        </div>
                    )}
                </div>
            )}
        </span>
    );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ImageDetailModal({
    images,
    initialIndex,
    debugMode: globalDebugMode,
    edgeThresholds: globalEdgeThresholds,
    overlayOpacity: globalOverlayOpacity,
    segmentationConf: globalSegConf,
    onClose,
    onAddToCart,
    cart,
}) {
    const [currentIndex, setCurrentIndex] = useState(initialIndex);
    const [localDebugMode, setLocalDebugMode] = useState(globalDebugMode || 'none');
    const [edgeLow, setEdgeLow] = useState(globalEdgeThresholds?.low ?? 50);
    const [edgeHigh, setEdgeHigh] = useState(globalEdgeThresholds?.high ?? 150);
    const [overlayOpacity, setOverlayOpacity] = useState(globalOverlayOpacity ?? 0.5);
    const [segConf, setSegConf] = useState(globalSegConf ?? 0.25);

    const [detail, setDetail] = useState(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState(null);

    const [imgLoading, setImgLoading] = useState(true);
    const [imgError, setImgError] = useState(false);

    const img = images[currentIndex] ?? null;

    // ── Fetch detail data when image changes ──────────────────────────────────
    useEffect(() => {
        if (!img) return;
        let cancelled = false;
        setDetail(null);
        setDetailLoading(true);
        setDetailError(null);

        api.get(`/images/${img.id}/detail`)
            .then(data => { if (!cancelled) setDetail(data); })
            .catch(err => { if (!cancelled) setDetailError(err?.message || 'Failed to load details'); })
            .finally(() => { if (!cancelled) setDetailLoading(false); });

        return () => { cancelled = true; };
    }, [img?.id]);

    // ── Reset image loading state when source changes ─────────────────────────
    useEffect(() => {
        setImgLoading(true);
        setImgError(false);
    }, [currentIndex, localDebugMode, edgeLow, edgeHigh, segConf]);

    // ── Lock body scroll ──────────────────────────────────────────────────────
    useEffect(() => {
        const prev = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        return () => { document.body.style.overflow = prev; };
    }, []);

    // ── Navigation ────────────────────────────────────────────────────────────
    const goNext = useCallback(() => setCurrentIndex(i => Math.min(i + 1, images.length - 1)), [images.length]);
    const goPrev = useCallback(() => setCurrentIndex(i => Math.max(i - 1, 0)), []);
    const toggleCart = useCallback(() => { if (img) onAddToCart(img.id); }, [img, onAddToCart]);

    // ── Keyboard shortcuts ────────────────────────────────────────────────────
    useEffect(() => {
        const handler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            switch (e.key) {
                case 'Escape':     onClose(); break;
                case 'ArrowLeft':  goPrev();  break;
                case 'ArrowRight': goNext();  break;
                case '1': setLocalDebugMode('none');         break;
                case '2': setLocalDebugMode('edges');        break;
                case '3': setLocalDebugMode('overlay');      break;
                case '4': setLocalDebugMode('depth');        break;
                case '5': setLocalDebugMode('complexity');   break;
                case '6': setLocalDebugMode('segmentation'); break;
                case '7': setLocalDebugMode('room');         break;
                case '8': setLocalDebugMode('materials');    break;
                case 'c': case 'C': toggleCart(); break;
                default: break;
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [onClose, goPrev, goNext, toggleCart]);

    // ── Derived values ────────────────────────────────────────────────────────
    const inCart = img ? cart.includes(img.id) : false;
    const currentMode = DEBUG_MODES.find(m => m.key === localDebugMode) ?? DEBUG_MODES[0];

    const mainSrc = buildDebugSrc(img, localDebugMode, edgeLow, edgeHigh, segConf);
    const edgesSrc = img ? `/api/v1/debug/images/${img.id}/edges?t1=${edgeLow}&t2=${edgeHigh}` : '';

    const filename = detail?.filename ?? img?.meta_data?.filename ?? (img ? `image_${img.id}` : '');

    // detail.tags is TagInfo[] from the API; img.tags is string[] from the grid search result (fallback).
    const tags = detail?.tags
        ?? (img?.tags ?? []).map(t => ({ label: t, source: 'preloaded', source_label: 'Imported with dataset' }));

    const groupedAttrs = useMemo(() => {
        if (!detail?.science_attributes?.length) return [];
        const groups = {};
        for (const attr of detail.science_attributes) {
            const prefix = attr.key.split('.')[0] || 'other';
            (groups[prefix] = groups[prefix] || []).push(attr);
        }
        return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b));
    }, [detail]);

    const hasScienceData = groupedAttrs.length > 0;
    const hasHumanData = (detail?.human_validations?.length ?? 0) > 0;

    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <div
            className="fixed inset-0 z-50 flex flex-col bg-black/80"
            role="dialog"
            aria-modal="true"
            aria-label={`Image detail: ${filename}`}
        >
            {/* Backdrop click to close */}
            <div className="absolute inset-0 -z-10" onClick={onClose} />

            {/* ── Header bar ────────────────────────────────────────────────── */}
            <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2.5 bg-gray-900 border-b border-gray-700 text-white min-w-0">
                {/* Prev */}
                <button
                    onClick={goPrev}
                    disabled={currentIndex === 0}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed text-sm transition-colors flex-shrink-0"
                    title="Previous image (←)"
                >
                    <ChevronLeft size={15} /> Prev
                </button>

                {/* Filename + position */}
                <div className="flex-1 min-w-0 flex items-center justify-center gap-3">
                    <span className="text-sm font-semibold truncate text-gray-100" title={filename}>
                        {filename}
                    </span>
                    <span className="text-xs text-gray-400 flex-shrink-0">
                        {currentIndex + 1} / {images.length}
                    </span>
                </div>

                {/* Next */}
                <button
                    onClick={goNext}
                    disabled={currentIndex === images.length - 1}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-gray-700 hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed text-sm transition-colors flex-shrink-0"
                    title="Next image (→)"
                >
                    Next <ChevronRight size={15} />
                </button>

                <div className="w-px h-5 bg-gray-600 mx-1 flex-shrink-0" />

                {/* Cart toggle */}
                <button
                    onClick={toggleCart}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors flex-shrink-0 ${
                        inCart
                            ? 'bg-blue-600 hover:bg-blue-700 text-white'
                            : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                    }`}
                    title="Toggle cart (C)"
                >
                    {inCart ? <Check size={15} /> : <ShoppingCart size={15} />}
                    {inCart ? 'In Cart' : 'Add to Cart'}
                </button>

                {/* Close */}
                <button
                    onClick={onClose}
                    className="p-2 rounded hover:bg-gray-700 text-gray-300 hover:text-white transition-colors flex-shrink-0"
                    title="Close (Esc)"
                >
                    <X size={17} />
                </button>
            </div>

            {/* ── Middle: image + right panel ───────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden min-h-0">

                {/* Image display area */}
                <div className="flex-1 flex items-center justify-center bg-gray-950 relative overflow-hidden min-w-0">
                    {img && (
                        <>
                            {localDebugMode === 'overlay' ? (
                                <div className="relative w-full h-full">
                                    <img
                                        src={img.url}
                                        alt={filename}
                                        className="w-full h-full object-contain block"
                                        onLoad={() => setImgLoading(false)}
                                        onError={() => { setImgLoading(false); setImgError(true); }}
                                    />
                                    <img
                                        src={edgesSrc}
                                        alt="Edge overlay"
                                        className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen"
                                        style={{ opacity: overlayOpacity }}
                                    />
                                </div>
                            ) : (
                                <img
                                    key={mainSrc}
                                    src={mainSrc}
                                    alt={filename}
                                    className={`w-full h-full object-contain transition-opacity duration-200 ${
                                        imgLoading ? 'opacity-0' : 'opacity-100'
                                    }`}
                                    onLoad={() => setImgLoading(false)}
                                    onError={() => { setImgLoading(false); setImgError(true); }}
                                />
                            )}

                            {imgLoading && !imgError && (
                                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                    <Loader2 className="animate-spin text-gray-500" size={40} />
                                </div>
                            )}

                            {imgError && !imgLoading && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 gap-2">
                                    <AlertCircle size={36} />
                                    <span className="text-sm">
                                        {localDebugMode !== 'none'
                                            ? `Debug mode "${currentMode.label}" unavailable for this image`
                                            : 'Image unavailable'}
                                    </span>
                                    {localDebugMode !== 'none' && (
                                        <button
                                            className="text-xs text-blue-400 hover:underline"
                                            onClick={() => setLocalDebugMode('none')}
                                        >
                                            Show original
                                        </button>
                                    )}
                                </div>
                            )}

                            {inCart && !imgError && (
                                <div className="absolute top-3 left-3 bg-blue-600 text-white text-xs font-semibold px-2 py-1 rounded flex items-center gap-1 shadow">
                                    <Check size={11} /> In Cart
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* ── Right panel: debug controls + tags + meta ─────────────── */}
                <div className="w-60 flex-shrink-0 bg-gray-900 border-l border-gray-700 flex flex-col overflow-y-auto text-white">

                    {/* Debug mode selector */}
                    <div className="p-3 border-b border-gray-700">
                        <div className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">
                            Debug Mode
                        </div>
                        <div className="space-y-0.5">
                            {DEBUG_MODES.map(mode => (
                                <label
                                    key={mode.key}
                                    className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded cursor-pointer transition-colors select-none ${
                                        localDebugMode === mode.key
                                            ? 'bg-blue-600 text-white'
                                            : 'text-gray-300 hover:bg-gray-800'
                                    }`}
                                >
                                    <input
                                        type="radio"
                                        name="debug_mode"
                                        value={mode.key}
                                        checked={localDebugMode === mode.key}
                                        onChange={() => setLocalDebugMode(mode.key)}
                                        className="sr-only"
                                    />
                                    <span className={`text-[10px] w-4 font-mono ${localDebugMode === mode.key ? 'text-blue-200' : 'text-gray-600'}`}>
                                        {mode.shortcut}
                                    </span>
                                    <span className="text-sm">{mode.label}</span>
                                </label>
                            ))}
                        </div>

                        {/* Mode-specific controls */}
                        {currentMode.hasEdgeSliders && (
                            <div className="mt-3 space-y-2.5 px-1">
                                <div>
                                    <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                                        <span>Low threshold</span>
                                        <span className="font-mono">{edgeLow}</span>
                                    </div>
                                    <input
                                        type="range" min={10} max={200} value={edgeLow}
                                        onChange={e => setEdgeLow(Number(e.target.value))}
                                        className="w-full accent-blue-400 h-1.5"
                                    />
                                </div>
                                <div>
                                    <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                                        <span>High threshold</span>
                                        <span className="font-mono">{edgeHigh}</span>
                                    </div>
                                    <input
                                        type="range" min={50} max={300} value={edgeHigh}
                                        onChange={e => setEdgeHigh(Number(e.target.value))}
                                        className="w-full accent-blue-400 h-1.5"
                                    />
                                </div>
                            </div>
                        )}

                        {currentMode.hasOverlaySlider && (
                            <div className="mt-2.5 px-1">
                                <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                                    <span>Overlay opacity</span>
                                    <span className="font-mono">{Math.round(overlayOpacity * 100)}%</span>
                                </div>
                                <input
                                    type="range" min={0} max={100}
                                    value={Math.round(overlayOpacity * 100)}
                                    onChange={e => setOverlayOpacity(Number(e.target.value) / 100)}
                                    className="w-full accent-blue-400 h-1.5"
                                />
                            </div>
                        )}

                        {currentMode.hasSegSlider && (
                            <div className="mt-3 px-1">
                                <div className="flex justify-between text-[10px] text-gray-400 mb-1">
                                    <span>Confidence</span>
                                    <span className="font-mono">{Math.round(segConf * 100)}%</span>
                                </div>
                                <input
                                    type="range" min={10} max={90}
                                    value={Math.round(segConf * 100)}
                                    onChange={e => setSegConf(Number(e.target.value) / 100)}
                                    className="w-full accent-blue-400 h-1.5"
                                />
                            </div>
                        )}

                        {!currentMode.hasEdgeSliders && !currentMode.hasSegSlider && localDebugMode !== 'none' && (
                            <p className="mt-2 text-[10px] text-gray-600 px-1">Model defaults — no controls.</p>
                        )}
                    </div>

                    {/* Tags */}
                    {tags.length > 0 && (
                        <div className="p-3 border-b border-gray-700">
                            <div className="flex items-center justify-between mb-2">
                                <div className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Tags</div>
                                <div className="flex items-center gap-2 text-[9px] text-gray-600">
                                    <span className="flex items-center gap-1">
                                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-500" /> imported
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-indigo-500" /> pipeline
                                    </span>
                                </div>
                            </div>
                            <div className="flex flex-wrap gap-1">
                                {tags.map(t => (
                                    <TagBadge key={`${t.label}-${t.source}`} tag={t} />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Image meta */}
                    {img && (
                        <div className="p-3 text-[10px] text-gray-500 space-y-1 flex-1">
                            <div>ID: <span className="font-mono text-gray-300">{img.id}</span></div>
                            {detail?.meta_data?.upload_batch_id && (
                                <div>Batch: <span className="font-mono text-gray-300 text-[9px]">{detail.meta_data.upload_batch_id}</span></div>
                            )}
                            <div className="pt-2 text-[9px] text-gray-700 leading-relaxed">
                                ← → navigate &nbsp;·&nbsp; 1–8 debug<br />
                                C cart &nbsp;·&nbsp; Esc close
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* ── Bottom panels: science attributes + human validations ────── */}
            <div className="flex-shrink-0 h-56 flex bg-white border-t border-gray-200 overflow-hidden">

                {/* Science attributes */}
                <div className="flex-1 flex flex-col min-w-0 border-r border-gray-200">
                    <div className="px-3 py-2 bg-gray-50 border-b border-gray-100 flex items-center justify-between flex-shrink-0">
                        <h3 className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">
                            Science Attributes
                        </h3>
                        <div className="flex items-center gap-2">
                            {detailLoading && <Loader2 className="animate-spin text-gray-400" size={12} />}
                            {!detailLoading && detail && (
                                <span className="text-[10px] text-gray-400">
                                    {detail.science_attributes.length} values · {groupedAttrs.length} groups
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-2">
                        {detailLoading && (
                            <div className="flex items-center justify-center h-full text-gray-400">
                                <Loader2 className="animate-spin mr-2" size={14} />
                                <span className="text-xs">Loading…</span>
                            </div>
                        )}

                        {detailError && !detailLoading && (
                            <div className="flex items-center gap-2 text-red-500 text-xs p-2">
                                <AlertCircle size={14} /> {detailError}
                            </div>
                        )}

                        {!detailLoading && !detailError && !hasScienceData && (
                            <p className="text-xs text-gray-400 italic p-2">
                                No science attributes yet. Run the pipeline on this image.
                            </p>
                        )}

                        {!detailLoading && !detailError && groupedAttrs.map(([prefix, attrs]) => (
                            <AttributeGroup
                                key={prefix}
                                prefix={prefix}
                                attrs={attrs}
                                defaultOpen={groupedAttrs.length <= 4}
                            />
                        ))}
                    </div>
                </div>

                {/* Human validations */}
                <div className="w-72 flex-shrink-0 flex flex-col">
                    <div className="px-3 py-2 bg-gray-50 border-b border-gray-100 flex items-center justify-between flex-shrink-0">
                        <h3 className="text-[10px] font-bold text-gray-600 uppercase tracking-widest">
                            Human Validations
                        </h3>
                        {!detailLoading && detail && (
                            <span className="text-[10px] text-gray-400">
                                {detail.human_validations.length} records
                            </span>
                        )}
                    </div>

                    <div className="flex-1 overflow-y-auto">
                        {detailLoading && (
                            <div className="flex items-center justify-center h-full text-gray-400">
                                <Loader2 className="animate-spin mr-2" size={14} />
                                <span className="text-xs">Loading…</span>
                            </div>
                        )}

                        {!detailLoading && !hasHumanData && !detailError && (
                            <p className="text-xs text-gray-400 italic p-3">
                                No human validations yet.
                            </p>
                        )}

                        {!detailLoading && detail?.human_validations?.map((v, i) => (
                            <div key={i} className="px-3 py-2 border-b border-gray-100 hover:bg-gray-50">
                                <div className="flex items-center justify-between mb-0.5">
                                    <span className="text-xs font-semibold text-gray-700 truncate">
                                        {v.username ?? (v.user_id != null ? `user_${v.user_id}` : 'Pipeline')}
                                    </span>
                                    <span className="text-xs font-mono font-bold text-blue-700 flex-shrink-0">
                                        {v.value.toFixed(2)}
                                    </span>
                                </div>
                                <div className="text-[10px] text-gray-500 font-mono truncate mb-0.5">
                                    {v.attribute_key}
                                </div>
                                <div className="flex gap-3 text-[10px] text-gray-400">
                                    {v.duration_ms != null && v.duration_ms > 0 && (
                                        <span>{v.duration_ms.toLocaleString()}ms</span>
                                    )}
                                    {v.created_at && (
                                        <span>{new Date(v.created_at).toLocaleDateString()}</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
