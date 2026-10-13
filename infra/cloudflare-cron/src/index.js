const REPO = "gonzaloiv/pressmirror";
const REF = "main";

const WORKFLOWS_BY_CRON = {
  "17 5 * * *": "daily-pipeline.yml",
  "23 6 * * *": "daily-pipeline.yml",
  "23 7 * * *": "publish-buffer.yml",
};

async function dispatchWorkflow(workflowFile, env) {
  const url =
    `https://api.github.com/repos/${REPO}/actions/workflows/` +
    `${workflowFile}/dispatches`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "pressmirror-cron",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: REF }),
  });
  if (response.status !== 204) {
    throw new Error(
      `Dispatch of ${workflowFile} failed: ${response.status} ` +
        `${await response.text()}`
    );
  }
}

export default {
  async scheduled(event, env, ctx) {
    const workflowFile = WORKFLOWS_BY_CRON[event.cron];
    if (!workflowFile) {
      console.log(`No workflow mapped to cron "${event.cron}"`);
      return;
    }
    ctx.waitUntil(dispatchWorkflow(workflowFile, env));
  },

  async fetch(request, env) {
    const workflowFile = new URL(request.url).searchParams.get("workflow");
    if (!workflowFile) {
      return new Response(
        "Usage: /?workflow=daily-pipeline.yml|publish-buffer.yml\n"
      );
    }
    try {
      await dispatchWorkflow(workflowFile, env);
      return new Response(`Dispatched ${workflowFile}\n`);
    } catch (error) {
      return new Response(`${error.message}\n`, { status: 502 });
    }
  },
};
