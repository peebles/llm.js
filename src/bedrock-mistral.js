import {
  BedrockRuntimeClient,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import debug from "debug";

// This templating system seems the closest to huggingface "tokenizer_config.json" files
import nunjucks from "nunjucks";
nunjucks.configure({ autoescape: false, trimBlocks: true, lstripBlocks: true });
const log = debug("llm.js:bedrock-mistral");

const MODEL = "mistral.mixtral-8x7b-instruct-v0:1";
// This is the prompt template for mistral
const PROMPT_TEMPLATE = `{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}`;

const BedrockOptions = [
  "max_tokens",
  "top_k",
  "top_p",
  "temperature",
  "stop",
];

// This works with mistral, and the default prompt.
const makePrompt = (messages, template) => {
  let _messages = [];
  for (let message of messages) {
    if (message.role === "system") {
      _messages.push({ role: "user", content: message.content });
      _messages.push({ role: "assistant", content: "ok" });
    } else if (message.role === "user") {
      _messages.push({ role: "user", content: message.content });
    } else if (message.role === "assistant") {
      _messages.push({ role: "assistant", content: message.content });
    } else {
      throw new Error("Only user and assistant roles are supported!");
    }
  }
  const context = {
    bos_token: "<s>",
    eos_token: "</s>",
    messages: _messages,
    raise_exception: (err) => { throw new Error(err); }
  };
  return nunjucks.renderString(template, context);
}

export default async function BedrockMistral(messages, options = {}) {
  if (!messages || messages.length === 0) { throw new Error("No messages provided") }

  const model = options.model || MODEL;
  const client = new BedrockRuntimeClient();

  let bedrockOptions = {};
  for (const key of BedrockOptions) {
    if (typeof options[key] !== "undefined") {
      bedrockOptions[key] = options[key];
    }
  }

  // build the prompt
  let prompt;
  if (options.makePrompt) {
    prompt = options.makePrompt(messages, options);
  } else if (options.promptTemplate) {
    prompt = makePrompt(messages, options.promptTemplate);
  } else {
    prompt = makePrompt(messages, PROMPT_TEMPLATE);
  }

  const params = {
    modelId: model,
    contentType: "application/json",
    accept: "application/json",
    body: JSON.stringify({
      prompt,
      ...bedrockOptions,
    }),
  };

  log(`sending to Bedrock Mistral with body ${JSON.stringify(params)}`);

  if (options.stream) {
    const response = await client.send(new InvokeModelWithResponseStreamCommand(params));
    return stream_response(response, options.usage);
  } else {
    const response = await client.send(new InvokeModelCommand(params));
    const rawRes = response.body;
    const jsonString = new TextDecoder().decode(rawRes);
    const parsedResponse = JSON.parse(jsonString);
    if (options.usage) {
      // The amazon-bedrock-invocationMetrics are apparently not available when not streaming,
      // so estimate them.
      let prompt_tokens = Math.floor(prompt.length / 4);
      let completion_tokens = Math.floor(parsedResponse.outputs[0].text.length / 4);
      await options.usage({
        prompt_tokens,
        completion_tokens,
      });
    }
    return parsedResponse.outputs[0].text
  }
}

BedrockMistral.defaultModel = MODEL;

export async function* stream_response(response, usage) {
  for await (const item of response.body) {
    const chunk = JSON.parse(new TextDecoder().decode(item.chunk.bytes));
    if (!(chunk && chunk.outputs && chunk.outputs[0])) continue;
    if (usage && chunk["amazon-bedrock-invocationMetrics"]) {
      const metrics = chunk["amazon-bedrock-invocationMetrics"];
      await usage({
        prompt_tokens: metrics.inputTokenCount,
        completion_tokens: metrics.outputTokenCount,
      });
      continue;
    }
    if (chunk.outputs[0].stop_reason) continue;
    yield chunk.outputs[0].text;
  }
}
