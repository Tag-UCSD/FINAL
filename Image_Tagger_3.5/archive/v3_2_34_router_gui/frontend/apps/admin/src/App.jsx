import React, { useState, useEffect } from 'react';
import { Header, Button, Toggle, ApiClient } from '@shared';
import { ShieldAlert, DollarSign, Server, Power, Info, RefreshCcw, Download } from 'lucide-react';

const api = new ApiClient('/api/v1/admin');

export default function AdminApp() {
    const [models, setModels] = useState([]);
    const [budget, setBudget] = useState(null);
    const [killSwitchActive, setKillSwitchActive] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [savingModelId, setSavingModelId] = useState(null);
    const [killBusy, setKillBusy] = useState(false);
    const [exportIds, setExportIds] = useState('');
    const [exportBusy, setExportBusy] = useState(false);
    const [exportMessage, setExportMessage] = useState(null);

    useEffect(() => {
        loadAll();
    }, []);

    async function loadAll() {
        setLoading(true);
        setError(null);
        try {
            const [modelsResp, budgetResp] = await Promise.all([
                api.get('/models'),
                api.get('/budget'),
            ]);
            setModels(Array.isArray(modelsResp) ? modelsResp : []);
            if (budgetResp) {
                setBudget(budgetResp);
                setKillSwitchActive(!!budgetResp.is_kill_switched);
            }
        } catch (err) {
            console.error('Failed to load admin data', err);
            setError(err.message || 'Failed to load admin cockpit');
        } finally {
            setLoading(false);
        }
    }

    async function handleToggle(id) {
        const current = models.find(m => m.id === id);
        if (!current) return;
        setSavingModelId(id);
        setError(null);
        try {
            const updated = await api.patch(`/models/${id}`, {
                is_enabled: !current.is_enabled,
            });
            setModels(models.map(m => (m.id === id ? updated : m)));
        } catch (err) {
            console.error('Failed to update model', err);
            setError(err.message || 'Failed to update model');
        } finally {
            setSavingModelId(null);
        }
    }

    async function handleCostBlur(id, rawValue) {
        const value = parseFloat(rawValue);
        if (Number.isNaN(value)) {
            return;
        }
        const current = models.find(m => m.id === id);
        if (!current || current.cost_per_1k_tokens === value) {
            return;
        }
        setSavingModelId(id);
        setError(null);
        try {
            const updated = await api.patch(`/models/${id}`, {
                cost_per_1k_tokens: value,
            });
            setModels(models.map(m => (m.id === id ? updated : m)));
        } catch (err) {
            console.error('Failed to update cost', err);
            setError(err.message || 'Failed to update model cost');
        } finally {
            setSavingModelId(null);
        }
    }

    async function handleKillSwitch(nextActive) {
        setKillBusy(true);
        setError(null);
        try {
            const resp = await api.post(`/kill-switch?active=${nextActive}
async function handleTrainingExport() {
    setExportBusy(true);
    setExportMessage(null);
    setError(null);
    try {
        const ids = exportIds
            .split(',')
            .map(s => parseInt(s.trim(), 10))
            .filter(n => !Number.isNaN(n));
        const payload = { image_ids: ids, format: 'json' };
        const data = await api.post('/training/export', payload);
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'training_export.json';
        a.click();
        URL.revokeObjectURL(url);
        setExportMessage(`Exported ${data.length || 0} rows to training_export.json`);
    } catch (err) {
        console.error('Training export failed', err);
        setError(err.message || 'Training export failed');
    } finally {
        setExportBusy(false);
    }
}
`);
            if (resp) {
                setBudget(resp);
                setKillSwitchActive(!!resp.is_kill_switched);
            }
        } catch (err) {
            console.error('Failed to toggle kill switch', err);
            setError(err.message || 'Failed to toggle kill switch');
        } finally {
            setKillBusy(false);
        }
    }

    const totalModels = models.length;
    const enabledModels = models.filter(m => m.is_enabled).length;

    const totalSpent = budget ? budget.total_spent : 0;
    const hardLimit = budget ? budget.hard_limit : 1;
    const usagePct = hardLimit > 0 ? Math.min(100, (totalSpent / hardLimit) * 100) : 0;

    return (
        <div className="min-h-screen bg-gray-100 pb-10">
            <Header appName="Admin" title="Cost & Governance Cockpit" />

            <div className="p-8 max-w-6xl mx-auto space-y-8">
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <p className="text-sm text-gray-500">
                            Configure which models and tools are allowed to run, and monitor budget risk.
                        </p>
                        {error && (
                            <p className="text-xs text-red-600 mt-1">
                                {error}
                            </p>
                        )}
                    </div>
                    <div className="flex items-center gap-3">
                        {loading && (
                            <span className="text-xs text-gray-500 flex items-center gap-2">
                                <RefreshCcw className="animate-spin" size={14} /> Loading…
                            </span>
                        )}
                        <Button variant="secondary" onClick={loadAll}>
                            <RefreshCcw size={16} className="mr-2" /> Refresh
                        </Button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Models & Tools */}
                    <section className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Server className="text-blue-500" size={18} />
                                <h2 className="font-semibold text-gray-900 text-sm">
                                    AI Models
                                </h2>
                            </div>
                            <span className="text-xs text-gray-400">
                                {enabledModels}/{totalModels} models enabled
                            </span>
                        </div>
                        <div className="divide-y divide-gray-100">
                            {models.map(model => (
                                <div
                                    key={model.id}
                                    className="flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
                                >
                                    <div>
                                        <p className="text-sm font-medium text-gray-900">
                                            {model.name}
                                        </p>
                                        <p className="text-xs text-gray-500">
                                            Provider: {model.provider || '—'}
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-right">
                                            <label className="block text-[11px] uppercase tracking-wide text-gray-500 font-semibold mb-1">
                                                Cost / 1K tokens
                                            </label>
                                            <div className="flex items-center gap-1">
                                                <DollarSign size={12} className="text-gray-400" />
                                                <input
                                                    type="number"
                                                    step="0.0001"
                                                    defaultValue={model.cost_per_1k_tokens}
                                                    onBlur={e => handleCostBlur(model.id, e.target.value)}
                                                    className="w-20 px-2 py-1 border border-gray-200 rounded text-xs text-right focus:outline-none focus:ring-1 focus:ring-blue-400"
                                                />
                                            </div>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <span className="text-[11px] text-gray-500 mb-1">
                                                {model.is_enabled ? 'Enabled' : 'Disabled'}
                                            </span>
                                            <Toggle
                                                checked={model.is_enabled}
                                                disabled={savingModelId === model.id}
                                                onChange={() => handleToggle(model.id)}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {!models.length && !loading && (
                                <div className="p-4 text-xs text-gray-400">
                                    No ToolConfigs found. Run the seed_tool_configs script or insert rows into
                                    the tool_configs table to populate this view.
                                </div>
                            )}
                        </div>
                    </section>

                    {/* Kill Switch & Budget */}
                    <section className="space-y-4">
                        <div className="bg-red-50 border border-red-100 rounded-xl p-4 flex flex-col gap-3">
                            <div className="flex items-center gap-2">
                                <ShieldAlert className="text-red-500" size={18} />
                                <h2 className="font-semibold text-sm text-red-800">
                                    Kill Switch
                                </h2>
                            </div>
                            <p className="text-xs text-red-700">
                                When activated, all paid models (cost_per_1k_tokens &gt; 0) are disabled. This is
                                enforced server-side via the ToolConfig table and checked before tools are used.
                            </p>
                            <div className="flex items-center justify-between mt-2">
                                <div>
                                    <p className="text-[11px] text-red-600 font-semibold uppercase tracking-wide">
                                        Status
                                    </p>
                                    <p className="text-sm font-medium text-red-900">
                                        {killSwitchActive ? 'ACTIVE' : 'Inactive'}
                                    </p>
                                </div>
                                <Button
                                    variant={killSwitchActive ? 'outline' : 'primary'}
                                    size="sm"
                                    disabled={killBusy}
                                    onClick={() => handleKillSwitch(!killSwitchActive)}
                                >
                                    <Power size={14} className="mr-1" />
                                    {killSwitchActive ? 'Disable Kill Switch' : 'Activate Kill Switch'}
                                </Button>
                            </div>
                            <p className="text-[11px] text-red-500 flex items-center gap-1 mt-1">
                                <Info size={12} /> Use this if cost monitoring shows you are approaching budget.
                            </p>
                        </div>

                        <div className="bg-slate-900 rounded-xl p-4 text-slate-50 shadow-sm">
<div className="bg-white rounded-xl border border-gray-200 p-4">
    <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
            <Download className="text-gray-600" size={18} />
            <h2 className="font-semibold text-sm text-gray-900">
                Training Export
            </h2>
        </div>
    </div>
    <p className="text-xs text-gray-500 mb-2">
        Export validated tags as JSON for fine-tuning or active learning. Provide a comma-separated
        list of image IDs (or leave blank to export nothing).
    </p>
    <textarea
        className="w-full text-xs border border-gray-200 rounded p-2 mb-2 focus:outline-none focus:ring-1 focus:ring-blue-400"
        rows={2}
        placeholder="e.g. 101, 102, 103"
        value={exportIds}
        onChange={e => setExportIds(e.target.value)}
    />
    <div className="flex items-center justify-between">
        <Button
            size="sm"
            variant="secondary"
            disabled={exportBusy}
            onClick={handleTrainingExport}
        >
            <Download size={14} className="mr-1" />
            Download JSON
        </Button>
        {exportMessage && (
            <span className="text-[11px] text-gray-500">
                {exportMessage}
            </span>
        )}
    </div>
</div>

                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <DollarSign className="text-emerald-300" size={18} />
                                    <h2 className="font-semibold text-sm">
                                        Cost Overview
                                    </h2>
                                </div>
                                <span className="text-[11px] text-slate-400">
                                    Prototype estimator
                                </span>
                            </div>
                            <div className="mt-4 space-y-1 text-xs">
                                <p>
                                    <span className="text-slate-400">Estimated spend:</span>{' '}
                                    <span className="font-semibold">${totalSpent.toFixed(2)}</span>
                                </p>
                                <p>
                                    <span className="text-slate-400">Hard limit:</span>{' '}
                                    <span className="font-semibold">${hardLimit.toFixed(2)}</span>
                                </p>
                            </div>
                            <div className="w-full bg-slate-800 h-2 rounded-full mt-4 overflow-hidden">
                                <div
                                    className="bg-emerald-300 h-full"
                                    style={{ width: `${usagePct}%` }}
                                />
                            </div>
                            <p className="text-[11px] text-slate-400 mt-2">
                                {usagePct.toFixed(0)}% of hard limit used.
                            </p>
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}