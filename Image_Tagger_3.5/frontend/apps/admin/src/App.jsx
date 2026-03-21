import React, { useState, useEffect } from 'react';
import { Header, Button, Toggle, ApiClient, useToast } from '@shared';
import { ShieldAlert, DollarSign, Server, Power, Info, RefreshCcw, Download, Activity } from 'lucide-react';

const api = new ApiClient('/api/v1/admin', { 'X-User-Role': 'admin' });
const vlmHealthApi = new ApiClient('/api/v1/vlm-health', { 'X-User-Role': 'admin' });


const VLMConfigPanel = () => {
  const toast = useToast();
  const [provider, setProvider] = React.useState('auto');
  const [engineName, setEngineName] = React.useState(null);
  const [available, setAvailable] = React.useState({});
  const [status, setStatus] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [imageId, setImageId] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [promptOverride, setPromptOverride] = React.useState('');
  const [maxBatchSize, setMaxBatchSize] = React.useState('');
  const [costPer1k, setCostPer1k] = React.useState('');

  const loadConfig = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await api.get('/vlm/config');
      if (resp) {
        setProvider(resp.provider || 'auto');
        setEngineName(resp.engine || null);
        setAvailable(resp.available_backends || {});
        setPromptOverride(resp.cognitive_prompt_override || '');
        setMaxBatchSize(
          typeof resp.max_batch_size === 'number' ? String(resp.max_batch_size) : ''
        );
        setCostPer1k(
          typeof resp.cost_per_1k_images_usd === 'number'
            ? String(resp.cost_per_1k_images_usd)
            : ''
        );
      }
    } catch (err) {
      console.error('Failed to load VLM config', err);
      setError(err.message || 'Failed to load VLM configuration');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    loadConfig();
  }, []);

  const handleSave = async () => {
    setBusy(true);
    setStatus(null);
    setError(null);
    try {
      await api.post('/vlm/config', {
        provider,
        cognitive_prompt_override: promptOverride || null,
        max_batch_size: maxBatchSize ? parseInt(maxBatchSize, 10) : null,
        cost_per_1k_images_usd: costPer1k ? parseFloat(costPer1k) : null,
      });
      setStatus('Saved VLM configuration.');
      toast.success('Saved VLM configuration.', { title: 'VLM Config' });
      await loadConfig();
    } catch (err) {
      console.error('Failed to save VLM config', err);
      setError(err.message || 'Failed to save VLM configuration');
      toast.error('Failed to save VLM configuration', { message: err.message || String(err) });
    } finally {
      setBusy(false);
    }
  };

  const handleTest = async () => {
    if (!imageId) {
      setStatus('Enter an image ID to test.');
      return;
    }
    setBusy(true);
    setStatus(null);
    setError(null);
    try {
      const payload = { image_id: Number(imageId), prompt: 'Quick VLM sanity check for this architectural image.' };
      const resp = await api.post('/vlm/test', payload);
      if (resp) {
        const label = resp.is_stub ? 'STUB (no API keys visible)' : 'LIVE';
        setStatus(`Test OK with ${resp.engine}: ${label}.`);
      } else {
        setStatus('No response from VLM test.');
      }
    } catch (err) {
      console.error('VLM test failed', err);
      setError(err.message || 'VLM test failed');
      toast.error('VLM test failed', { message: err.message || String(err) });
    } finally {
      setBusy(false);
    }
  };

  const providerLabel = (key) => {
    switch (key) {
      case 'auto':
        return 'Auto (prefer Gemini, then OpenAI, then Anthropic)';
      case 'gemini':
        return 'Gemini (Google, e.g. 1.5 Flash/Pro)';
      case 'openai':
        return 'OpenAI (e.g. GPT-4o / 4.1)';
      case 'anthropic':
        return 'Anthropic (e.g. Claude 3.5)';
      case 'stub':
        return 'Stub (no network calls, neutral outputs)';
      default:
        return key;
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex flex-col">
          <h2 className="font-semibold text-sm text-gray-900">VLM Engine</h2>
          <p className="text-[11px] text-gray-500">
            Choose which Visual Language Model to use for cognitive analysis.
          </p>
        </div>
        {loading && (
          <span className="text-[11px] text-gray-400">Loading…</span>
        )}
      </div>


<div className="mt-2 p-2 rounded-md bg-blue-50 border border-blue-100 text-[11px] text-blue-900 space-y-1">
  <p className="font-semibold">VLM configuration: handle with care</p>
  <ul className="list-disc ml-4 space-y-0.5">
    <li>Changes here affect which provider (Gemini, OpenAI, Anthropic, stub) is used for all VLM-assisted analyses.</li>
    <li>Always verify that API keys are detected and that the effective engine looks correct before enabling real runs.</li>
    <li>After editing settings, run a small test on a known image ID and confirm the response looks plausible.</li>
    <li>If tests fail or output looks broken, revert to a safe configuration and notify an engineer.</li>
  </ul>
</div>

      <div className="space-y-3">
        <div className="space-y-1">
          <label className="text-[11px] uppercase tracking-wide text-gray-500 font-semibold">
            Provider
          </label>
          <select
            className="w-full text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            disabled={busy}
          >
            <option value="auto">Auto (prefer Gemini)</option>
            <option value="gemini">Gemini</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="stub">Stub</option>
          </select>
          <p className="text-[11px] text-gray-400">
            {providerLabel(provider)}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[11px] text-gray-500">
          <div>
            <span className="font-semibold text-gray-700">Detected keys:</span>
            <ul className="mt-1 space-y-0.5">
              <li>Gemini: {available.gemini ? 'yes' : 'no'}</li>
              <li>OpenAI: {available.openai ? 'yes' : 'no'}</li>
              <li>Anthropic: {available.anthropic ? 'yes' : 'no'}</li>
            </ul>
          </div>
          <div>
            <span className="font-semibold text-gray-700">Effective engine:</span>
            <p className="mt-1 text-gray-800 text-xs">
              {engineName || 'Stub / none configured'}
            </p>
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-[11px] uppercase tracking-wide text-gray-500 font-semibold">
            Test on Image ID
          </label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              className="w-24 text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-400"
              placeholder="e.g. 1"
              value={imageId}
              onChange={(e) => setImageId(e.target.value)}
            />
            <Button
              size="xs"
              variant="secondary"
              onClick={handleTest}
              disabled={busy}
            >
              Test VLM
            </Button>
          </div>
        </div>

        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-2">
            <Button
              size="xs"
              variant="primary"
              onClick={handleSave}
              disabled={busy}
            >
              Save
            </Button>
          </div>
          <div className="flex-1 text-right">
            {status && (
              <p className="text-[11px] text-gray-600">{status}</p>
            )}
            {error && (
              <p className="text-[11px] text-red-600">{error}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};


const BulkUploadPanel = ({ onUploadCompleted }) => {
  const toast = useToast();
  const [files, setFiles] = React.useState(null);
  const [status, setStatus] = React.useState(null);
  const [isUploading, setIsUploading] = React.useState(false);

  const handleChange = (e) => {
    setFiles(e.target.files);
  };

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setStatus("Please choose one or more image files first.");
      return;
    }
    setIsUploading(true);
    setStatus("Uploading...");
    try {
      const form = new FormData();
      Array.from(files).forEach((file) => form.append("files", file));
      const res = await fetch("/api/v1/admin/upload", {
        method: "POST",
        headers: {
          "X-User-Role": "admin",
        },
        body: form,
      });
      if (!res.ok) {
        const text = await res.text();
        setStatus(`Upload failed: ${res.status} ${text}`);
        toast.error(`Upload failed: ${res.status}`, { title: 'Bulk Upload' });
        setIsUploading(false);
        return;
      }
      const data = await res.json();
      setStatus(
        `Uploaded ${data.created_count} images. New IDs: ${data.image_ids.join(", ")}`
      );
      toast.success(`Uploaded ${data.created_count} images.`, { title: 'Bulk Upload' });

      if (onUploadCompleted) {
        try {
          onUploadCompleted(data);
        } catch (callbackErr) {
          console.error('BulkUploadPanel onUploadCompleted callback failed', callbackErr);
        }
      }
    } catch (err) {
      setStatus(`Upload error: ${String(err)}`);
      toast.error('Upload error', { message: String(err) });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="mt-6 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-900">Bulk image upload</h2>
      <p className="mt-1 text-sm text-gray-600">
        Use this when you want to add new images to the dataset without touching the
        command line. Files will be stored under IMAGE_STORAGE_ROOT and made
        available to the science pipeline and Explorer.
      </p>
      <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center">
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleChange}
          className="text-sm"
        />
        <button
          type="button"
          onClick={handleUpload}
          disabled={isUploading}
          className="inline-flex items-center rounded-md border border-transparent bg-blue-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>
      </div>
      {status && (
        <p className="mt-2 text-xs text-gray-700 whitespace-pre-wrap">{status}</p>
      )}
    </div>
  );
};



const UploadJobsPanel = ({ jobs, loading, onRefresh, lastJobId }) => {
    const hasJobs = Array.isArray(jobs) && jobs.length > 0;

    return (
        <div>
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Server size={16} className="text-slate-500" />
                    <h2 className="font-semibold text-sm text-gray-900">
                        Upload jobs
                    </h2>
                </div>
                <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => onRefresh && onRefresh()}
                    disabled={loading}
                >
                    <RefreshCcw
                        size={14}
                        className={loading ? 'animate-spin mr-1' : 'mr-1'}
                    />
                    <span className="text-[11px]">Refresh</span>
                </Button>
            </div>

            {lastJobId && (
                <p className="text-[11px] text-gray-500 mb-1">
                    Last upload job:{' '}
                    <span className="font-mono text-gray-800">#{lastJobId}</span>
                </p>
            )}

            {!hasJobs && (
                <p className="text-[11px] text-gray-500">
                    No upload jobs recorded yet. Upload a batch to see progress here.
                </p>
            )}

            {hasJobs && (
                <div className="mt-2 max-h-56 overflow-y-auto text-xs">
                    <table className="w-full text-left">
                        <thead className="text-[11px] text-gray-500 border-b border-gray-200">
                            <tr>
                                <th className="py-1 pr-2">Job</th>
                                <th className="py-1 pr-2">Status</th>
                                <th className="py-1 pr-2">Progress</th>
                                <th className="py-1">Errors</th>
                            </tr>
                        </thead>
                        <tbody>
                            {jobs.map(job => (
                                <tr key={job.id} className="border-b border-gray-100">
                                    <td className="py-1 pr-2 font-mono text-[11px]">
                                        #{job.id}
                                    </td>
                                    <td className="py-1 pr-2">
                                        <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium bg-slate-100 text-slate-700">
                                            {job.status}
                                        </span>
                                    </td>
                                    <td className="py-1 pr-2">
                                        {job.completed_items}/{job.total_items}
                                    </td>
                                    <td
                                        className="py-1 text-[10px] text-gray-500 truncate max-w-[120px]"
                                        title={job.error_summary || ''}
                                    >
                                        {job.failed_items > 0
                                            ? `${job.failed_items} failed`
                                            : '—'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

function CostHistoryCard({ dailyCosts, totalSpent, hardLimit }) {
    if (!dailyCosts || dailyCosts.length === 0) {
        return (
            <div className="bg-white rounded-xl border border-gray-200 p-4">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <DollarSign className="text-emerald-500" size={18} />
                        <h2 className="font-semibold text-sm text-gray-900">
                            Cost history
                        </h2>
                    </div>
                </div>
                <p className="text-[11px] text-gray-500">
                    No VLM usage has been recorded yet. Once the science pipeline
                    calls an external model, a daily cost history will appear here.
                </p>
            </div>
        );
    }

    const maxCost = Math.max(
        ...dailyCosts.map(p => (typeof p.total_cost === 'number' ? p.total_cost : 0)),
        0.01
    );
    const lastSeven = dailyCosts.slice(-7);
    const prevSeven = dailyCosts.slice(-14, -7);

    const sum = arr =>
        arr.reduce(
            (acc, p) => acc + (typeof p.total_cost === 'number' ? p.total_cost : 0),
            0
        );

    const lastSevenTotal = sum(lastSeven);
    const prevSevenTotal = sum(prevSeven);
    const delta = lastSevenTotal - prevSevenTotal;

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <DollarSign className="text-emerald-500" size={18} />
                    <h2 className="font-semibold text-sm text-gray-900">
                        Cost history
                    </h2>
                </div>
                <span className="text-[11px] text-gray-500">
                    Last {dailyCosts.length} days
                </span>
            </div>

            <div className="flex items-baseline justify-between text-xs mb-3">
                <div>
                    <div className="text-gray-500">Last 7 days</div>
                    <div className="font-semibold text-gray-900">
                        ${lastSevenTotal.toFixed(2)}
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-gray-500">Change vs. prior 7 days</div>
                    <div
                        className={
                            delta >= 0 ? 'text-red-600 font-semibold' : 'text-emerald-600 font-semibold'
                        }
                    >
                        {delta >= 0 ? '+' : ''}
                        {delta.toFixed(2)}
                    </div>
                </div>
            </div>

            <div className="flex items-end gap-0.5 h-16 mb-2">
                {dailyCosts.map((point, idx) => {
                    const value =
                        typeof point.total_cost === 'number' ? point.total_cost : 0;
                    const heightPct = Math.max(8, (value / maxCost) * 100);
                    return (
                        <div
                            key={idx}
                            className="flex-1 bg-emerald-100"
                            style={{ height: `${heightPct}%` }}
                            title={`${point.day}: $${value.toFixed(2)}`}
                        />
                    );
                })}
            </div>

            <div className="text-[11px] text-gray-500 flex justify-between mt-1">
                <span>Total spent: ${totalSpent.toFixed(2)}</span>
                <span>Budget: ${hardLimit.toFixed(2)}</span>
            </div>
        </div>
    );
}



function VLMHealthCard() {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;

        async function fetchRuns() {
            setLoading(true);
            setError(null);
            try {
                const data = await vlmHealthApi.get('/runs');
                if (!cancelled) {
                    setRuns(Array.isArray(data) ? data : []);
                }
            } catch (err) {
                console.error('Failed to load VLM health runs', err);
                if (!cancelled) {
                    setError('Unable to load VLM health runs.');
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        }

        fetchRuns();
        return () => {
            cancelled = true;
        };
    }, []);

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Activity className="text-indigo-500" size={18} />
                    <h2 className="font-semibold text-sm text-gray-900">
                        VLM health runs
                    </h2>
                </div>
                {loading && (
                    <span className="text-[11px] text-gray-400">Loading…</span>
                )}
            </div>

            {error && (
                <p className="text-[11px] text-red-600 mb-1">
                    {error}
                </p>
            )}

            {!loading && (!runs || runs.length === 0) && (
                <p className="text-[11px] text-gray-500">
                    No VLM health runs found yet. After you run{' '}
                    <code className="bg-gray-100 px-1 py-[1px] rounded text-[10px]">
                        make vlm-health-init
                    </code>{' '}
                    and the follow-up steps from the VLM Health Quickstart,
                    runs will appear here.
                </p>
            )}

            {runs && runs.length > 0 && (
                <div className="space-y-2 mt-1">
                    {runs.map((run) => (
                        <div
                            key={run.run_id}
                            className="flex items-center justify-between text-[11px]"
                        >
                            <div className="flex flex-col">
                                <span className="font-mono text-xs text-gray-900">
                                    {run.run_id}
                                </span>
                                {run.created_at && (
                                    <span className="text-[10px] text-gray-500">
                                        {new Date(run.created_at).toLocaleString()}
                                    </span>
                                )}
                            </div>
                            <div className="flex flex-col items-end gap-1">
                                <div className="flex gap-2">
                                    <span
                                        className={
                                            'px-1.5 py-[1px] rounded-full text-[10px] ' +
                                            (run.has_variance_audit
                                                ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                                                : 'bg-gray-50 text-gray-400 border border-gray-100')
                                        }
                                    >
                                        variance
                                    </span>
                                    <span
                                        className={
                                            'px-1.5 py-[1px] rounded-full text-[10px] ' +
                                            (run.has_turing_summary
                                                ? 'bg-blue-50 text-blue-700 border border-blue-100'
                                                : 'bg-gray-50 text-gray-400 border border-gray-100')
                                        }
                                    >
                                        turing
                                    </span>
                                </div>
                                <div className="flex gap-2">
                                    {run.has_variance_audit && (
                                        <a
                                            href={`/api/v1/vlm-health/runs/${encodeURIComponent(
                                                run.run_id
                                            )}/variance-audit`}
                                            className="text-[10px] text-blue-600 hover:underline"
                                        >
                                            CSV
                                        </a>
                                    )}
                                    {run.has_turing_summary && (
                                        <a
                                            href={`/api/v1/vlm-health/runs/${encodeURIComponent(
                                                run.run_id
                                            )}/turing-summary`}
                                            className="text-[10px] text-blue-600 hover:underline"
                                        >
                                            Summary
                                        </a>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <p className="mt-3 text-[10px] text-gray-400">
                This view is read-only. Edit or regenerate runs from the CLI
                using the VLM Health SOP.
            </p>
        </div>
    );
}

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
    const [imageExportBusy, setImageExportBusy] = useState(false);
    const [imageExportMessage, setImageExportMessage] = useState(null);

    const [dailyCosts, setDailyCosts] = useState([]);
    const [costWindowDays, setCostWindowDays] = useState(30);
    const [uploadJobs, setUploadJobs] = useState([]);
    const [uploadJobsLoading, setUploadJobsLoading] = useState(false);
    const [lastUploadJobId, setLastUploadJobId] = useState(null);
    useEffect(() => {
        loadAll();
        loadUploadJobs();
    }, []);

    async function loadAll() {
        setLoading(true);
        setError(null);
        try {
            const [modelsResp, budgetResp, costsResp] = await Promise.all([
                api.get('/models'),
                api.get('/budget'),
                api.get(`/costs/daily?days=${costWindowDays}`),
            ]);
            setModels(Array.isArray(modelsResp) ? modelsResp : []);
            if (budgetResp) {
                setBudget(budgetResp);
                setKillSwitchActive(!!budgetResp.is_kill_switched);
                setDailyCosts(Array.isArray(costsResp) ? costsResp : []);
            }
        } catch (err) {
            console.error('Failed to load admin data', err);
            setError(err.message || 'Failed to load admin cockpit');
        } finally {
            setLoading(false);
        }
    }

async function loadUploadJobs(limit = 20) {
    setUploadJobsLoading(true);
    try {
        const jobs = await api.get(`/upload/jobs?limit=${limit}`);
        setUploadJobs(Array.isArray(jobs) ? jobs : []);
    } catch (err) {
        console.error('Failed to load upload jobs', err);
        // We treat this as a soft failure to avoid breaking the main admin load.
    } finally {
        setUploadJobsLoading(false);
    }
}

function handleUploadCompleted(data) {
    if (data && typeof data.job_id === 'number') {
        setLastUploadJobId(data.job_id);
    }
    // Refresh the job list so progress is visible shortly after upload.
    loadUploadJobs();
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
            const resp = await api.post(`/kill-switch?active=${nextActive}`);
            if (resp) {
                // Backend returns BudgetStatus; keep UI in sync.
                setBudget(resp);
                setKillSwitchActive(!!resp.is_kill_switched);
            }
        } catch (err) {
            console.error('Failed to update kill switch', err);
            setError(
                err && err.message
                    ? err.message
                    : 'Failed to update kill switch'
            );
        } finally {
            setKillBusy(false);
        }
    }

    async function handleTrainingExport() {
        setExportBusy(true);
        setExportMessage(null);
        try {
            const ids = exportIds
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean)
                .map((s) => Number(s))
                .filter((n) => !Number.isNaN(n));

            const payload = { image_ids: ids };

            const resp = await fetch('/api/v1/admin/training/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-Role': 'admin',
                },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const text = await resp.text();
                setExportMessage(`Export failed: ${resp.status} ${text}`);
                return;
            }

            const data = await resp.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json',
            });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'training_export.json';
            a.click();
            window.URL.revokeObjectURL(url);

            setExportMessage(
                `Exported ${Array.isArray(data) ? data.length : 0} training examples.`
            );
        } catch (err) {
            console.error('Training export failed', err);
            setExportMessage(
                `Training export failed: ${err && err.message ? err.message : String(err)}`
            );
        } finally {
            setExportBusy(false);
        }
    }

    async function handleImageExport() {
        setImageExportBusy(true);
        setImageExportMessage(null);
        try {
            const resp = await fetch('/api/v1/admin/export/images', {
                method: 'GET',
                headers: {
                    'X-User-Role': 'admin',
                },
            });

            if (!resp.ok) {
                const text = await resp.text();
                setImageExportMessage(`Export failed: ${resp.status} ${text}`);
                return;
            }

            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'image_tagger_images_export.zip';
            a.click();
            window.URL.revokeObjectURL(url);

            setImageExportMessage('Image export started. Check your downloads.');
        } catch (err) {
            console.error('Image export failed', err);
            setImageExportMessage(
                `Image export failed: ${err && err.message ? err.message : String(err)}`
            );
        } finally {
            setImageExportBusy(false);
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

<div className="mt-1 mb-2 p-2 rounded-md bg-blue-50 border border-blue-100 text-[11px] text-blue-900 space-y-1">
    <p className="font-semibold">What this export is for</p>
    <ul className="list-disc ml-4 space-y-0.5">
        <li>Each row in the exported JSON corresponds to an image/case with its validated tags.</li>
        <li>Use this for training downstream models (BNs, regressions, or fine-tuned vision models).</li>
        <li>Keep track of which canon version and schema were active when you generated the export.</li>
        <li>For reproducible studies, store export parameters (image IDs, filters) alongside your analysis code.</li>
    </ul>
</div>
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

    <div className="bg-white rounded-xl border border-gray-200 p-4 mt-4">
        <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
                <Download className="text-gray-600" size={18} />
                <h2 className="font-semibold text-sm text-gray-900">
                    Image Archive Export
                </h2>
            </div>
        </div>
        <p className="text-xs text-gray-500 mb-2">
            Download a zip of all stored image files for offline analysis or backup.
            This uses the same storage paths as the science pipeline.
        </p>
        <div className="flex items-center justify-between">
            <Button
                size="sm"
                variant="secondary"
                disabled={imageExportBusy}
                onClick={handleImageExport}
            >
                <Download size={14} className="mr-1" />
                Download .zip
            </Button>
            {imageExportMessage && (
                <span className="text-[11px] text-gray-500">
                    {imageExportMessage}
                </span>
            )}
        </div>
    </div>

    {/* Admin Tools: Bulk Upload & VLM */}
    <div className="mt-4 grid grid-cols-1 gap-4">
        <div className="bg-white rounded-xl border border-gray-200 p-4">
            <BulkUploadPanel onUploadCompleted={handleUploadCompleted} />
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
            <VLMConfigPanel />
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
            <UploadJobsPanel
                jobs={uploadJobs}
                loading={uploadJobsLoading}
                onRefresh={loadUploadJobs}
                lastJobId={lastUploadJobId}
            />
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
                        <div className="mt-4">
                            <CostHistoryCard
                                dailyCosts={dailyCosts}
                                totalSpent={totalSpent}
                                hardLimit={hardLimit}
                            />
                        </div>
                    </section>
                </div>
            </div>
        </div>
    );
}