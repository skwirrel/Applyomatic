import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import OpenAI from 'openai';
import { fileURLToPath } from 'node:url';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { inspect } from 'node:util';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { marked } from 'marked';
import TerminalRenderer from 'marked-terminal';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ————————————————————————————————————————————————
// Configuration helpers
// ————————————————————————————————————————————————
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const TEMPERATURE = parseFloat(process.env.OPENAI_TEMPERATURE || '0.3');

/**
 * Validates that required environment variables are set.
 * Throws an error if any required variable is missing.
 */
function assertEnv() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set. Create a .env file.');
  }
}

/**
 * Reads a JSON file from the given path and parses it into an object.
 * @param {string} p - Path to the JSON file.
 * @returns {Promise<Object>} Parsed JSON object.
 */
async function readJSON(p) {
  const s = await fs.readFile(p, 'utf8');
  return JSON.parse(s);
}

/**
 * Reads a text file from the given path and returns its content as a string.
 * @param {string} p - Path to the text file.
 * @returns {Promise<string>} File content as a string.
 */
async function readText(p) {
  return fs.readFile(p, 'utf8');
}

/**
 * Generates a random integer between the specified minimum and maximum values (inclusive).
 * @param {number} min - Minimum value.
 * @param {number} max - Maximum value.
 * @returns {number} Random integer.
 */
function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Returns the current date and time in ISO 8601 format.
 * @returns {string} Current date and time in ISO format.
 */
function isoNow() {
  return new Date().toISOString();
}

// ————————————————————————————————————————————————
// LLM wrapper using Responses API with structured outputs
// ————————————————————————————————————————————————
/**
 * Sends a request to the LLM (Language Model) with the specified parameters.
 * Supports structured outputs using JSON Schema.
 * @param {Object} params - Parameters for the LLM request.
 * @param {string} params.instructions - Instructions for the LLM.
 * @param {string} params.input - Input text for the LLM.
 * @param {Object} [params.schema] - JSON Schema for structured output.
 * @param {number} [params.temperature] - Temperature for the LLM (default is TEMPERATURE).
 * @returns {Promise<any>} Response from the LLM.
 */
async function sendToLLM({ instructions, input, schema, temperature = TEMPERATURE }) {
  const options = {
    model: MODEL,
    tools: [{
      type: "web_search_preview",
      search_context_size: "medium",
      user_location: {
        "type": "approximate",
        "country": "GB",
      }
    }],
    temperature,
    instructions: instructions || undefined,
    input,
  };

  if (schema) {
    options.text = {
      format: {
        name: 'structured_output',
        type: 'json_schema',
        schema,
        strict: true,
      },
    };
  }

  if (process.env.DEBUG === 'true') {
    console.log('DEBUG: Sending to LLM:', inspect(options, { depth: null, colors: true }));
  }

  const res = await client.responses.create(options);

  if (process.env.DEBUG === 'true') {
    console.log('DEBUG: Response from LLM:', JSON.stringify(res, null, 2));
  }

  return schema ? JSON.parse(res.output_text) : res.output_text;
}

// ————————————————————————————————————————————————
// Domain logic — all functions now return structured outputs
// ————————————————————————————————————————————————
/**
 * Fetches the job description for the specified job title.
 * Returns the job description and sources consulted.
 * @param {string} job - Job title or description.
 * @returns {Promise<Object>} Job description and sources.
 */
async function fetchJobDescription(job) {
  const input = [
    {
      role: 'user',
      content: `Find the current job description for: "${job}". Return concise text and any sources consulted. If multiple postings exist, pick the most authoritative and most recent.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      role_title: { type: 'string' },
      job_description: { type: 'string' },
      sources: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            title: { type: 'string' },
            url: { type: 'string' },
            publisher: { type: 'string' },
          },
         required: ['url','title','publisher']
        }
      }
    },
    required: ['role_title','job_description','sources']
  };
  const result = await sendToLLM({ input, schema, instructions: 'Extract and condense faithfully; include sources when possible.' });
  return result; // { role_title?, job_description, sources? }
}

/**
 * Checks whether applications are still open for the specified job.
 * Returns a boolean indicating the status and a confidence score.
 * @param {string} job - Job title or description.
 * @returns {Promise<Object>} Application status and confidence score.
 */
async function applicationStillOpen(job) {
  const input = [
    {
      role: 'user',
      content: `Check whether applications are currently open for: "${job}". Today is ${new Date().toISOString()}.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      open: { type: 'boolean' },
      confidence: { type: 'number', minimum: 0, maximum: 1 },
    },
    required: ['open', 'confidence',]
  };
  return sendToLLM({ input, schema, instructions: 'Return a careful, up-to-date assessment.' });
}

/**
 * Assesses the relevance of a skill to the specified job description.
 * Returns a score and a rationale for the relevance.
 * @param {string} skillDescription - Description of the skill.
 * @param {string} jobDescription - Job description.
 * @returns {Promise<Object>} Relevance score and rationale.
 */
async function assessSkillRelevanceToNewJob(skillDescription, jobDescription) {
  const input = [
    { role: 'system', content: 'Return JSON only.' },
    {
      role: 'user',
      content: `
I am considering applying for the following job. Here is the dob description:
=============================
${jobDescription}
=============================

Please can you rate the relevance of the following skill for this job:
=============================
${skillDescription}
=============================
Please provide a score from 1 (not very relevant) to 10 (highly relevant) and provide a one-sentence rationale.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      score: { type: 'integer', minimum: 1, maximum: 10 },
      rationale: { type: 'string' }
    },
    required: ['score','rationale']
  };
  return sendToLLM({ input, schema }); // { score, rationale? }
}

/**
 * Assesses the relevance of a past role to a new job based on the provided job description.
 *
 * This function sends a structured input to a language model to evaluate how relevant a past role
 * from a CV is to a given job description. The language model returns a score between 1 and 10,
 * along with a one-sentence rationale for the score.
 *
 * @async
 * @function
 * @param {Object} role - The past role to assess.
 * @param {string} role.jobTitle - The title of the past role.
 * @param {string} role.from - The start date of the past role.
 * @param {string} role.to - The end date of the past role.
 * @param {string} [role.description] - A description of the past role (optional).
 * @param {string} jobDescription - The description of the new job being considered.
 * @returns {Promise<Object>} A promise that resolves to an object containing:
*/
async function assessRoleRelevanceToNewJob(role, jobDescription) {
  const input = [
    { role: 'system', content: 'Return JSON only.' },
    {
      role: 'user',
      content: `Job description:
I am considering applying for the following job. Here is the dob description:
=============================
${jobDescription}
=============================

Please can you assess the relevance of the following past role from my CV to this job:
=============================
Job Title: ${role.jobTitle}
From: ${role.from} To: ${role.to}
Description: ${role.description || '(no description provided)'}
=============================
Please provide a score from 1 (not very relevant) to 10 (highly relevant) and provide a one-sentence rationale.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      score: { type: 'integer', minimum: 1, maximum: 10 },
      rationale: { type: 'string' }
    },
    required: ['score','rationale']
  };
  return sendToLLM({ input, schema }); // { score, rationale? }
}

function byScoreDesc(a, b) { return b.score - a.score; }
function byStartDateDesc(a, b) { return (b.from || '').localeCompare(a.from || ''); }

/**
 * Converts structured CV data into a well-formatted Markdown document.
 * 
 * Input must include `personalDetails`, `qualifications`, `skills`, and `pastJobRoles`.
 * The CV structure should already be job-tailored (e.g. skills sorted by relevance).
 * 
 * This function sends the JSON-formatted CV to an OpenAI model using the Responses API,
 * and expects structured output including the final Markdown content.
 * 
 * @param {Object} cv - Tailored CV data as a JavaScript object
 * @returns {Promise<Object>} - Structured LLM response including `cv_markdown`
 */
 async function composeCVMarkdown(cv) {
  const cvJSON = JSON.stringify(cv, null, 2);
  const input = [
    {
      role: 'user',
      content: `Here is my CV as structured data (JSON):
=============================
${cvJSON}
=============================
Please convert this to a clean, well-formatted Markdown CV suitable for UK applications. Include dates as in dd/mm/yyyy format.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      cv_markdown: { type: 'string' },
    },
    required: ['cv_markdown']
  };
  return sendToLLM({ input, schema, instructions: 'Write professional CV Markdown.' }); // { cv_markdown, headings? }
}

/**
 * Requests detailed suggestions for improving a CV in Markdown format.
 *
 * The function sends the entire Markdown CV to an OpenAI model via the Responses API
 * and returns a structured list of edits. Each suggestion includes a type,
 * an optional location within the document, and a free-form text recommendation.
 *
 * Suggested edit types may include: 'clarity', 'typo', 'formatting', 'impact',
 * 'consistency', 'tone', or 'other'.
 *
 * @param {string} cvMarkdown - The CV in Markdown format
 * @returns {Promise<Object>} - An object with a `suggestions` array of edit proposals
 */
async function suggestCVImprovements(cvMarkdown) {
  const input = [
    {
      role: 'user',
      content: `Here is my current CV (Markdown). Please suggest specific improvements, corrections, or formatting fixes.

=============================
${cvMarkdown}
=============================`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      suggestions: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', enum: ['clarity', 'typo', 'formatting', 'impact', 'consistency', 'tone', 'other'] },
            location: { type: 'string', description: 'Heading or brief excerpt' },
            suggestion: { type: 'string' }
          },
          required: ['type', 'location', 'suggestion']
        }
      }
    },
    required: ['suggestions']
  };
  return sendToLLM({ input, schema, instructions: 'Provide actionable, concise suggestions.' });
}

/**
 * Applies a list of approved CV edits to the original Markdown document using the LLM.
 *
 * Sends both the original Markdown CV and a list of structured, user-approved suggestions
 * to the OpenAI model. The LLM is instructed to apply only those edits, making no additional
 * changes unless strictly necessary to implement them. The result is returned as updated Markdown.
 *
 * This function uses structured output (JSON Schema) to ensure a predictable response format.
 *
 * @param {string} cvMarkdown - The original CV in Markdown format
 * @param {Array<Object>} suggestions - Approved edits, each with `type`, `location`, and `suggestion`
 * @returns {Promise<Object>} - Object containing `cv_markdown` (the revised CV)
 */
 async function applyCVEdits(cvMarkdown, suggestions) {
  const input = [
    {
      role: 'user',
      content: `Revise the CV applying ONLY the approved edits below.

Original CV:
=============================
${cvMarkdown}
=============================

Approved edits (JSON):
=============================
${JSON.stringify(suggestions, null, 2)}
=============================
Return final edited Markdown only in the schema.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      cv_markdown: { type: 'string' }
    },
    required: ['cv_markdown']
  };
  return sendToLLM({
    input,
    schema,
    instructions:
      'Apply edits faithfully; do not introduce new content unless necessary to implement an edit.',
  });
}

// —— interactive helpers ——
function ask(rl, q, defaultInput = '') {
   rl.write(defaultInput);
   return rl.question(q, { defaultInput });
}

function printSuggestions(suggestions) {
  console.log('\nSuggested edits:');
  suggestions.forEach((s, i) => {
    const loc = s.location ? ` @ ${s.location}` : '';
    console.log(`${i + 1}. [${s.type}]${loc}\n   ${s.suggestion}\n`);
  });
}

function parseIndexList(inputStr, max) {
  const set = new Set();
  for (const part of inputStr.split(/\s*,\s*/)) {
    if (!part) continue;
    if (part.includes('-')) {
      const [a, b] = part.split('-').map((n) => parseInt(n, 10));
      if (Number.isInteger(a) && Number.isInteger(b)) {
        const start = Math.min(a, b);
        const end = Math.max(a, b);
        for (let i = start; i <= end; i += 1) if (i >= 1 && i <= max) set.add(i - 1);
      }
    } else {
      const n = parseInt(part, 10);
      if (Number.isInteger(n) && n >= 1 && n <= max) set.add(n - 1);
    }
  }
  return Array.from(set).sort((a, b) => a - b);
}

const TYPE_OPTIONS = ['clarity', 'typo', 'formatting', 'impact', 'consistency', 'tone', 'other'];

/**
 * Interactively reviews and modifies a list of suggestions.
 * 
 * This function provides a command-line interface for users to review,
 * edit, or delete suggestions from a given list. Users can interactively
 * modify the list using commands such as delete, edit, continue, or quit.
 * 
 * @async
 * @function
 * @param {Array<Object>} suggestions - An array of suggestion objects to review.
 * @param {string} suggestions[].suggestion - The suggestion text.
 * @returns {Promise<Array<Object>>} A promise that resolves to the modified list of suggestions.
 */
async function reviewSuggestionsInteractively(suggestions) {

  const rl = readline.createInterface({ input, output });
  let list = suggestions.map((s) => ({ ...s }));

  try {
    for (;;) {
      printSuggestions(list);
      console.log('Commands: d <idx[,..]|a> delete | e <idx> edit | c continue | q quit');
      const line = (await rl.question('> ')).trim();
      if (!line) continue;

      const [cmd, restRaw] = line.split(/\s+/, 2);
      const rest = restRaw || '';

      if (cmd === 'q') return [];          // user aborted
      if (cmd === 'c') return list;        // user happy → apply all once

      if (cmd === 'd') {
        if (rest === 'a') { list = []; continue; }
        const idxs = parseIndexList(rest, list.length);
        if (!idxs.length) { console.log('No valid indices.'); continue; }
        list = list.filter((_, i) => !idxs.includes(i));
        continue;
      }

      if (cmd === 'e') {
        const idx = parseInt(rest, 10) - 1;
        if (!Number.isInteger(idx) || idx < 0 || idx >= list.length) {
          console.log('Out of range.');
          continue;
        }
        const item = { ...list[idx] };

      console.log(`\nEditing #${idx + 1}`);
      console.log(`Current suggestion: ${item.suggestion}`);
      const newSug = (await ask(rl, 'Enter new suggestion (leave blank to keep current): ', item.suggestion)).trim();
      if (newSug) item.suggestion = newSug;

        list[idx] = item;
        continue;
      }

      console.log('Unknown command.');
    }
  } finally {
    rl.close();
  }
}

/**
 * Enhances a CV written in Markdown format by providing suggestions for improvements
 * and allowing the user to interactively review and apply those suggestions.
 *
 * @async
 * @function polishCV
 * @param {string} cvMarkdown - The CV content in Markdown format.
 * @returns {Promise<string>} - A promise that resolves to the improved CV in Markdown format.
 */
async function polishCV(cvMarkdown) {

  marked.setOptions({
    renderer: new TerminalRenderer(),
  });
  console.log('\nPreview of your CV in Markdown format:\n=================================================================\n');
  console.log(marked(cvMarkdown));

  const { suggestions } = await suggestCVImprovements(cvMarkdown);
  if (!suggestions || suggestions.length === 0) return cvMarkdown;

  const approvedList = await reviewSuggestionsInteractively(suggestions);
  if (!approvedList.length) return cvMarkdown; // user quit or removed everything

  const { cv_markdown } = await applyCVEdits(cvMarkdown, approvedList);
  return cv_markdown;
}

/**
 * Generates a persuasive and tailored covering letter for a job application.
 *
 * @async
 * @function draftCoveringLetter
 * @param {Object} cvBaseData - The applicant's CV data, containing background information.
 * @param {string} coveringLetterThoughts - Personal notes to guide the tone and emphasis of the letter.
 * @param {string} jobDescription - The job description for the position being applied for.
 * @returns {Promise<Object>} A promise that resolves to an object containing the generated covering letter.
 * @throws {Error} If the input data is invalid or the letter generation fails.
 */
async function draftCoveringLetter(cvBaseData, coveringLetterThoughts, jobDescription) {
  const input = [
    {
      role: 'user',
      content: `I am applying for this job:
=============================
${jobDescription}
=============================

Here is some background about me (CV data):
=============================
${JSON.stringify(cvBaseData, null, 2)}
=============================

Here are personal notes to guide tone and emphasis:
=============================
${coveringLetterThoughts}
=============================

Write a persuasive, tailored covering letter in British English, 400–650 words.`,
    },
  ];
  const schema = {
    type: 'object',
    additionalProperties: false,
    properties: {
      covering_letter: { type: 'string' },
    },
    required: ['covering_letter']
  };
  return sendToLLM({ input, schema, instructions: 'Produce a concise, persuasive letter aligned to the role.' });
}

// ————————————————————————————————————————————————
// Orchestrator
// ————————————————————————————————————————————————
/**
 * Drafts a CV tailored to the specified job description.
 * Filters and ranks skills, achievements, and past roles based on relevance.
 * @param {Object} cvBaseData - Base CV data.
 * @param {string} jobDescription - Job description.
 * @returns {Promise<string>} Drafted CV in Markdown format.
 */
async function draftCV(cvBaseData, jobDescription) {
  const cv = {
    personalDetails: cvBaseData.personalDetails,
    qualifications: cvBaseData.qualifications,
  };

// Skills with relevance
console.log(`[INFO] Assessing relevance of skills to the job description.`);
const skillsWithRelevance = [];
for (const skill of cvBaseData.skills || []) {
   console.log(`[INFO] Assessing skill: "${skill}"`);
   const { score, rationale } = await assessSkillRelevanceToNewJob(skill, jobDescription);
   console.log(`[INFO] Skill "${skill}" scored ${score}/10. Rationale: ${rationale}`);
   skillsWithRelevance.push({ skill, score, rationale });
}
skillsWithRelevance.sort(byScoreDesc);
cv.skills = skillsWithRelevance.slice(0, 10).map(s => s.skill);
console.log(`[INFO] Top 10 relevant skills selected.`);

// Achievements with relevance
console.log(`[INFO] Assessing relevance of achievements to the job description.`);
const achievementsWithRelevance = [];
for (const achievement of cvBaseData.achievements || []) {
   console.log(`[INFO] Assessing achievement: "${achievement}"`);
   const { score, rationale } = await assessSkillRelevanceToNewJob(achievement, jobDescription);
   console.log(`[INFO] Achievement "${achievement}" scored ${score}/10. Rationale: ${rationale}`);
   achievementsWithRelevance.push({ achievement, score, rationale });
}
achievementsWithRelevance.sort(byScoreDesc);
cv.achievements = achievementsWithRelevance.slice(0, 10).map(a => a.achievement);
console.log(`[INFO] Top 10 relevant achievements selected.`);

// Roles with relevance
console.log(`[INFO] Assessing relevance of past job roles to the job description.`);
const rolesToInclude = [];
for (const role of cvBaseData.pastJobRoles || []) {
   console.log(`[INFO] Assessing role: "${role.jobTitle}" (${role.from} - ${role.to || 'Present'})`);
   const { score } = await assessRoleRelevanceToNewJob(role, jobDescription);
   console.log(`[INFO] Role "${role.jobTitle}" scored ${score}/10.`);
   const includeRole = { ...role };
   if (score < 80) {
      console.log(`[INFO] Role "${role.jobTitle}" is less relevant. Omitting description.`);
      delete includeRole.description;
   }
   rolesToInclude.push(includeRole);
}
rolesToInclude.sort(byStartDateDesc);
cv.pastJobRoles = rolesToInclude;
console.log(`[INFO] Relevant past job roles selected and sorted by start date.`);

  const { cv_markdown } = await composeCVMarkdown(cv);
  const polished = await polishCV(cv_markdown);
  return polished;
}

async function runOnce({ job, cvBasePath, notesPath, jobCfg }) {
  console.log(`[INFO] Starting job application process for: "${job}"`);

  console.log(`[INFO] Reading covering letter notes from: ${notesPath}`);
  const coveringLetterThoughts = await readText(notesPath);

  console.log(`[INFO] Reading CV base data from: ${cvBasePath}`);
  const cvBaseData = await readJSON(cvBasePath);

  console.log(`[INFO] Fetching job description for: "${job}"`);
  const jobInfo = await fetchJobDescription(job);
  console.log(`[INFO] Job description fetched successfully.`);

  console.log(`[INFO] Checking if applications are still open for: "${job}"`);
  const openStatus = await applicationStillOpen(job);
  if (!openStatus.open) {
    console.log(`[WARN] Applications are closed for: "${job}"`);
    console.log(`[INFO] Checked at: ${openStatus.checked_at_utc}; Confidence: ${openStatus.confidence ?? 'n/a'}`);
    return { gotJob: false, nextRun: null, jobDescription: jobInfo.job_description };
  }
  console.log(`[INFO] Applications are still open.`);

  console.log(`[INFO] Drafting CV tailored to the job description.`);
  const cv = await draftCV(cvBaseData, jobInfo.job_description);
  console.log(`[INFO] CV drafted successfully.`);

  console.log(`[INFO] Drafting covering letter tailored to the job description.`);
  const { covering_letter } = await draftCoveringLetter(cvBaseData, coveringLetterThoughts, jobInfo.job_description);
  console.log(`[INFO] Covering letter drafted successfully.`);

  console.log(`
COVERING LETTER:
==============================
${covering_letter}
==============================
`);
  console.log(`CV (Markdown):
==============================
${cv}
==============================
`);

  const gotJob = false; // Placeholder for human submission outcome

  let nextRun = null;
  if (!gotJob) {
    const days = randInt(jobCfg.minReapplyDays || 30, jobCfg.maxReapplyDays || 90);
    nextRun = new Date(Date.now() + days * 24 * 60 * 60 * 1000);
    console.log(`[INFO] Will try again in ~${days} days, around ${nextRun.toISOString()}`);
  }
  return { gotJob, nextRun, jobDescription: jobInfo.job_description };
}

async function main() {
  console.log(`[INFO] Validating environment variables.`);
  assertEnv();

  console.log(`[INFO] Parsing command-line arguments.`);
  const argv = yargs(hideBin(process.argv))
    .option('job', { type: 'string', desc: 'Job title / description to target' })
    .option('cv', { type: 'string', default: path.join(__dirname, '..', 'config', 'cv.base.json'), desc: 'Path to CV base JSON' })
    .option('notes', { type: 'string', default: path.join(__dirname, '..', 'config', 'covering-letter-notes.example.md'), desc: 'Path to covering letter notes (markdown/text)' })
    .option('jobcfg', { type: 'string', default: path.join(__dirname, '..', 'config', 'job.config.json'), desc: 'Path to job runtime config' })
    .option('once', { type: 'boolean', default: false, desc: 'Run once and exit' })
    .option('daemon', { type: 'boolean', default: false, desc: 'Run continuously with waits between attempts' })
    .help()
    .parse();

  console.log(`[INFO] Reading job configuration from: ${argv.jobcfg}`);
  const jobCfg = await readJSON(argv.jobcfg);

  const job = argv.job || jobCfg.job;
  if (!job) throw new Error('[ERROR] No job specified. Provide --job or set in config/job.config.json');

  let attempt = 0;
  const maxAttempts = jobCfg.maxAttempts ?? 3;

  const loop = async () => {
    attempt += 1;
    console.log(`
—— Attempt ${attempt} at ${isoNow()} ——`);
    const res = await runOnce({ job, cvBasePath: argv.cv, notesPath: argv.notes, jobCfg });
    if (res.gotJob) {
      console.log(`[SUCCESS] Job secured. Exiting.`);
      process.exit(0);
    }
    if (attempt >= maxAttempts || argv.once || !argv.daemon) {
      console.log(`[INFO] Exiting. (Either run once, attempts exhausted, or daemon not enabled.)`);
      process.exit(0);
    }
    const waitMs = res.nextRun ? res.nextRun.getTime() - Date.now() : 45 * 24 * 3600 * 1000;
    const safeMs = Math.max(waitMs, 1000);
    console.log(`[INFO] Sleeping for ${Math.round(safeMs / 1000)} seconds…`);
    setTimeout(loop, safeMs);
  };

  if (argv.daemon || jobCfg.daemon) {
    console.log(`[INFO] Running in daemon mode.`);
    await loop();
  } else {
    console.log(`[INFO] Running once.`);
    await runOnce({ job, cvBasePath: argv.cv, notesPath: argv.notes, jobCfg });
  }
}

// Run
main().catch((err) => {
  console.error(`[ERROR] ${err.message}`);
  process.exit(1);
});
