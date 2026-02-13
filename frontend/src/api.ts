const baseUrl = (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000";

async function handleResponse(res: Response) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${baseUrl}/health`);
  return handleResponse(res);
}

export async function getModelReport() {
  const res = await fetch(`${baseUrl}/reports/latest`);
  return handleResponse(res);
}

export async function getValidationReport() {
  const res = await fetch(`${baseUrl}/reports/validation/latest`);
  return handleResponse(res);
}

export async function trainUpload(file: File, target?: string, problemType?: string) {
  const form = new FormData();
  form.append("file", file);
  if (target) form.append("target", target);
  if (problemType) form.append("problem_type", problemType);
  const res = await fetch(`${baseUrl}/train/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function predictUpload(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${baseUrl}/predict/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export async function predictJson(records: any[]) {
  const res = await fetch(`${baseUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records }),
  });
  return handleResponse(res);
}
