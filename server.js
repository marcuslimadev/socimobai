import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

loadEnvFile();

const app = express();
app.use(express.json({ limit: '1mb' }));

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3100);
const API_KEY = String(process.env.SOCIMOB_AI_API_KEY || '').trim();
const AI_PROVIDER = String(process.env.AI_PROVIDER || 'auto').trim().toLowerCase();
const OLLAMA_BASE_URL = String(process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434').replace(/\/$/, '');
const DEFAULT_MODEL = String(process.env.OLLAMA_MODEL || 'llama3.1:8b');
const POLLINATIONS_BASE_URL = String(process.env.POLLINATIONS_BASE_URL || 'https://text.pollinations.ai/openai').replace(/\/$/, '');
const POLLINATIONS_MODEL = String(process.env.POLLINATIONS_MODEL || 'openai');
const POLLINATIONS_API_KEY = String(process.env.POLLINATIONS_API_KEY || '').trim();
const MAX_HISTORY_CHARS = Number(process.env.MAX_HISTORY_CHARS || 6000);
const MAX_PROPERTIES = Number(process.env.MAX_PROPERTIES || 8);
const TRAINING_FILE = String(
  process.env.SOCIMOB_TRAINING_FILE || path.join(__dirname, 'data', 'treinamento_imobiliaria.jsonl'),
);
const MAX_TRAINING_EXAMPLES = Number(process.env.MAX_TRAINING_EXAMPLES || 5);

const training = loadTraining(TRAINING_FILE);

function loadEnvFile(filePath = '.env') {
  try {
    if (!fs.existsSync(filePath)) return;

    const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;

      const separator = trimmed.indexOf('=');
      if (separator <= 0) continue;

      const key = trimmed.slice(0, separator).trim();
      let value = trimmed.slice(separator + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"'))
        || (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      process.env[key] = value;
    }
  } catch (error) {
    console.warn(`Could not load env file ${filePath}: ${error.message}`);
  }
}

function requireAuth(req, res, next) {
  if (!API_KEY) return next();

  const token = String(req.headers.authorization || '').replace(/^Bearer\s+/i, '').trim();
  if (token !== API_KEY) {
    return res.status(401).json({ success: false, error: 'Unauthorized' });
  }

  return next();
}

function clampText(value, maxChars) {
  const text = String(value || '').trim();
  if (text.length <= maxChars) return text;
  return text.slice(-maxChars);
}

function normalizeText(value) {
  return String(value || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function loadTraining(filePath) {
  const result = {
    file: filePath,
    loaded: false,
    systemRules: [],
    examples: [],
    error: null,
  };

  try {
    if (!fs.existsSync(filePath)) {
      result.error = `Arquivo de treinamento nao encontrado: ${filePath}`;
      return result;
    }

    const lines = fs.readFileSync(filePath, 'utf8')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    lines.forEach((line, index) => {
      try {
        const parsed = JSON.parse(line);
        const messages = Array.isArray(parsed?.messages) ? parsed.messages : [];
        const system = messages.find((message) => message?.role === 'system' && message?.content)?.content;
        const user = messages.find((message) => message?.role === 'user' && message?.content)?.content;
        const assistant = [...messages].reverse()
          .find((message) => message?.role === 'assistant' && message?.content)?.content;

        if (system && !result.systemRules.includes(system)) result.systemRules.push(system);
        if (assistant) {
          result.examples.push({
            index: index + 1,
            user: user || '',
            assistant,
            search: normalizeText(`${user || ''} ${assistant}`),
          });
        }
      } catch (error) {
        result.error = `Linha ${index + 1} invalida no treinamento: ${error.message}`;
      }
    });

    result.loaded = result.examples.length > 0 || result.systemRules.length > 0;
    return result;
  } catch (error) {
    result.error = error.message;
    return result;
  }
}

function selectTrainingExamples(body) {
  if (!training.examples.length || MAX_TRAINING_EXAMPLES <= 0) return [];

  const query = normalizeText([
    body.message,
    body.history,
    body.lead?.objetivo_compra,
    body.lead?.preferencia_bairro,
    body.lead?.localizacao,
  ].filter(Boolean).join(' '));

  const terms = new Set(query.split(' ').filter((term) => term.length >= 4));

  return training.examples
    .map((example) => {
      const score = example.search.split(' ')
        .filter((term) => terms.has(term))
        .length;
      return { ...example, score };
    })
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, MAX_TRAINING_EXAMPLES);
}

function trainingExamplesBlock(body) {
  const examples = selectTrainingExamples(body);
  if (!examples.length) return 'Nenhum exemplo de treinamento carregado.';

  return examples.map((example) => [
    `Exemplo ${example.index}:`,
    example.user ? `Cliente: ${example.user}` : null,
    `Resposta modelo: ${clampText(example.assistant, 900)}`,
  ].filter(Boolean).join('\n')).join('\n\n');
}

function propertyLine(property) {
  const bedrooms = Number(property?.dormitorios || 0) + Number(property?.suites || 0);
  const rent = Number(property?.valor_aluguel || 0);
  const sale = Number(property?.valor_venda || 0);
  const price = rent > 0
    ? `aluguel R$ ${rent.toLocaleString('pt-BR')}`
    : sale > 0
      ? `venda R$ ${sale.toLocaleString('pt-BR')}`
      : 'valor sob consulta';

  return [
    `Codigo ${property?.codigo_imovel || property?.codigo || 'N/A'}`,
    property?.tipo_imovel || 'Imovel',
    [property?.bairro, property?.cidade].filter(Boolean).join(', ') || 'local nao informado',
    `${bedrooms || '?'} quartos`,
    price,
  ].join(' | ');
}

function buildPrompt(body) {
  const assistantName = body.assistant_name || 'Teresa';
  const companyName = body.company_name || 'Exclusiva Lar Imoveis';
  const properties = Array.isArray(body.properties) ? body.properties.slice(0, MAX_PROPERTIES) : [];
  const lead = body.lead && typeof body.lead === 'object' ? body.lead : {};
  const missing = [];

  if (!lead.objetivo_compra) missing.push('compra ou aluguel');
  if (!lead.localizacao && !lead.preferencia_bairro) missing.push('bairro ou regiao');
  if (!lead.budget_max && !lead.budget_min) missing.push('faixa de valor');
  if (!lead.quartos) missing.push('quantidade de quartos');
  if (!lead.prazo_compra) missing.push('prazo');
  if (!lead.renda_mensal) missing.push('renda aproximada');

  const propertyContext = properties.length
    ? properties.map((item) => `- ${propertyLine(item)}`).join('\n')
    : 'Nenhum imovel real foi enviado nesta chamada.';

  const leadContext = Object.entries(lead)
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => `- ${key}: ${value}`)
    .join('\n') || 'Nenhum dado estruturado do lead.';

  return {
    system: `${body.system_prompt || ''}

BASE DE TREINAMENTO SOCIMOB:
${training.systemRules.join('\n') || 'Use atendimento imobiliario consultivo, empatico e orientado a proximo passo.'}

REGRAS OPERACIONAIS SOCIMOB:
- Voce e ${assistantName}, da ${companyName}.
- Responda sempre em portugues do Brasil.
- Seja educada, consultiva e acolhedora.
- Nao seja seca, nao pressione e nao repita pergunta respondida.
- Use o estilo dos exemplos treinados: conduza o atendimento, explique o proximo passo e, quando fizer sentido, sugira exemplos curtos de resposta.
- Faca no maximo uma pergunta por mensagem.
- Nao invente imoveis: use somente os imoveis reais abaixo.
- Exemplos de imoveis no treinamento sao apenas referencia de estilo; nunca trate como estoque real.
- Se o cliente der codigo, data, bairro, quartos, renda ou prazo, reconheca antes de avancar.
- Se faltar informacao, peca apenas o primeiro item faltante.
- Nao diga que e IA ou robo.
- Limite normal: ate ${Number(body.max_words || 120)} palavras.`,
    user: `HISTORICO:
${clampText(body.history, MAX_HISTORY_CHARS)}

DADOS DO LEAD:
${leadContext}

DADOS FALTANTES:
${missing.length ? missing.join(', ') : 'nenhum dado principal pendente'}

IMOVEIS REAIS DISPONIVEIS:
${propertyContext}

EXEMPLOS RELEVANTES DO TREINAMENTO:
${trainingExamplesBlock(body)}

MENSAGEM ATUAL DO CLIENTE:
${body.message || ''}

Responda como ${assistantName}:`,
  };
}

async function ollamaChat(body) {
  const prompt = buildPrompt(body);
  const model = body.ollama_model || DEFAULT_MODEL;

  const response = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      messages: [
        { role: 'system', content: prompt.system },
        { role: 'user', content: prompt.user },
      ],
      options: {
        temperature: 0.35,
        num_predict: 220,
      },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Ollama ${response.status}: ${text.slice(0, 500)}`);
  }

  const data = await response.json();
  const content = data?.message?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('Resposta invalida do Ollama');
  }

  return { content: content.trim(), model };
}

async function pollinationsChat(body) {
  const prompt = buildPrompt(body);
  const model = body.pollinations_model || POLLINATIONS_MODEL;
  const headers = { 'Content-Type': 'application/json' };
  if (POLLINATIONS_API_KEY) headers.Authorization = `Bearer ${POLLINATIONS_API_KEY}`;

  const response = await fetch(POLLINATIONS_BASE_URL, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model,
      stream: false,
      temperature: 0.45,
      max_tokens: 260,
      messages: [
        { role: 'system', content: prompt.system },
        { role: 'user', content: prompt.user },
      ],
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Pollinations ${response.status}: ${text.slice(0, 500)}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content || data?.message?.content || data?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('Resposta invalida da Pollinations');
  }

  return { content: content.trim(), model, provider: 'pollinations' };
}

function localTrainedReply(body) {
  const lead = body.lead && typeof body.lead === 'object' ? body.lead : {};
  const message = String(body.message || '').trim();
  const name = lead.nome || lead.name || lead.cliente || '';
  const firstName = String(name).split(/\s+/).filter(Boolean)[0] || '';
  const prefix = firstName ? `${firstName}, ` : '';

  const examples = selectTrainingExamples(body);
  const best = examples[0];
  const lower = normalizeText(message);

  if (/\b(oi|ola|bom dia|boa tarde|boa noite)\b/.test(lower)) {
    return {
      content: `${prefix}que bom falar com voce. Me conte, por favor: voce procura comprar, alugar ou vender um imovel? Exemplo: "quero alugar apartamento ate R$ 1.500".`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.objetivo_compra) {
    return {
      content: `${prefix}entendi. Para eu te conduzir certinho, voce procura compra, aluguel ou venda? Pode responder, por exemplo: "quero alugar" ou "quero comprar".`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.localizacao && !lead.preferencia_bairro) {
    return {
      content: `${prefix}perfeito. Qual bairro ou regiao voce prefere? Se estiver aberto a opcoes, pode me dizer algo como "pode ser perto do centro" ou "qualquer bairro ate esse valor".`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.budget_max && !lead.budget_min) {
    return {
      content: `${prefix}otimo, ja anotei a regiao. Qual faixa de valor voce quer considerar? Exemplo: "ate R$ 1.500" para aluguel ou "ate R$ 300 mil" para compra.`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.quartos) {
    return {
      content: `${prefix}certo. Quantos quartos seriam ideais para voce? Pode responder "1 quarto", "2 quartos" ou "1 ou 2", por exemplo.`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.prazo_compra) {
    return {
      content: `${prefix}ja tenho o principal para filtrar melhor. Para quando voce pretende se mudar ou concluir essa etapa? Exemplo: "este mes", "em 15 dias" ou "sem pressa".`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (best?.assistant) {
    return {
      content: clampText(best.assistant, 700),
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  return {
    content: `${prefix}perfeito, obrigado pelas informacoes. Vou seguir com uma busca mais alinhada ao que voce precisa e, se alguma opcao fizer sentido, posso te mostrar os detalhes ou chamar um corretor para continuar com voce.`,
    model: 'local-trained-rules',
    provider: 'local-trained',
  };
}

async function generateChat(body) {
  const errors = [];

  if (AI_PROVIDER === 'ollama') {
    const result = await ollamaChat(body);
    return { ...result, provider: 'ollama' };
  }

  if (AI_PROVIDER === 'pollinations') {
    return pollinationsChat(body);
  }

  if (AI_PROVIDER === 'local') {
    return localTrainedReply(body);
  }

  try {
    const result = await ollamaChat(body);
    return { ...result, provider: 'ollama' };
  } catch (error) {
    errors.push(`ollama: ${error.message}`);
  }

  try {
    return await pollinationsChat(body);
  } catch (error) {
    errors.push(`pollinations: ${error.message}`);
  }

  const fallback = localTrainedReply(body);
  return {
    ...fallback,
    fallback: true,
    warnings: errors,
  };
}

app.get('/health', requireAuth, async (_req, res) => {
  const trainingHealth = {
    loaded: training.loaded,
    file: training.file,
    examples: training.examples.length,
    rules: training.systemRules.length,
    error: training.error,
  };

  if (AI_PROVIDER === 'pollinations') {
    return res.json({
      success: true,
      provider: 'pollinations',
      model: POLLINATIONS_MODEL,
      training: trainingHealth,
    });
  }

  if (AI_PROVIDER === 'local') {
    return res.json({
      success: true,
      provider: 'local-trained',
      model: 'local-trained-rules',
      training: trainingHealth,
    });
  }

  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    const data = response.ok ? await response.json() : null;
    const models = Array.isArray(data?.models) ? data.models.map((item) => item.name).filter(Boolean) : [];

    return res.json({
      success: response.ok,
      provider: AI_PROVIDER === 'auto' ? 'auto' : 'ollama',
      model: DEFAULT_MODEL,
      model_available: models.includes(DEFAULT_MODEL),
      models,
      pollinations_fallback: AI_PROVIDER === 'auto',
      local_fallback: AI_PROVIDER === 'auto',
      training: trainingHealth,
    });
  } catch (error) {
    if (AI_PROVIDER === 'auto') {
      return res.json({
        success: true,
        provider: 'auto',
        model: DEFAULT_MODEL,
        ollama_available: false,
        pollinations_fallback: true,
        local_fallback: true,
        training: trainingHealth,
        warning: error.message,
      });
    }

    return res.status(503).json({
      success: false,
      provider: 'ollama',
      model: DEFAULT_MODEL,
      training: trainingHealth,
      error: error.message,
    });
  }
});

app.post('/chat', requireAuth, async (req, res) => {
  try {
    const result = await generateChat(req.body || {});
    return res.json({
      success: true,
      content: result.content,
      model: result.model,
      provider: result.provider || 'socimob-ai-gateway',
      fallback: result.fallback || false,
      warnings: result.warnings || undefined,
    });
  } catch (error) {
    return res.status(502).json({
      success: false,
      error: error.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Socimob AI Gateway listening on port ${PORT}`);
  if (training.loaded) {
    console.log(`Socimob training loaded: ${training.examples.length} examples from ${training.file}`);
  } else {
    console.warn(`Socimob training not loaded: ${training.error || 'no examples'}`);
  }
});
