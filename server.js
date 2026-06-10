import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
loadEnvFile(path.join(__dirname, 'runtime.env'));
loadEnvFile(path.join(__dirname, '.env'));

const app = express();
app.use(express.json({ limit: '1mb' }));

const PORT = Number(process.env.PORT || 3100);
const API_KEY = String(process.env.SOCIMOB_AI_API_KEY || '').trim();
const AI_PROVIDER = String(process.env.AI_PROVIDER || 'auto').trim().toLowerCase();
const OLLAMA_BASE_URL = String(process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434').replace(/\/$/, '');
const DEFAULT_MODEL = String(process.env.OLLAMA_MODEL || 'llama3.1:8b');
const HF_ROUTER_BASE_URL = String(process.env.HF_ROUTER_BASE_URL || 'https://router.huggingface.co/v1').replace(/\/$/, '');
const HF_TOKEN = String(process.env.HF_TOKEN || process.env.HUGGINGFACE_TOKEN || '').trim();
const HF_MODEL = String(process.env.HF_MODEL || 'Qwen/Qwen2.5-7B-Instruct:fastest');
const LOCAL_MODEL_BASE_URL = String(process.env.LOCAL_MODEL_BASE_URL || '').replace(/\/$/, '');
const LOCAL_MODEL_API_KEY = String(process.env.LOCAL_MODEL_API_KEY || '').trim();
const LOCAL_MODEL_NAME = String(process.env.LOCAL_MODEL_NAME || 'socimobai-finetuned');
const LOCAL_MODEL_TIMEOUT_MS = Number(process.env.LOCAL_MODEL_TIMEOUT_MS || 45000);
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

function hasRealProperties(body) {
  return Array.isArray(body?.properties) && body.properties.length > 0;
}

function combinedConversationText(body) {
  const lead = body?.lead && typeof body.lead === 'object' ? body.lead : {};

  return normalizeText([
    body?.message,
    body?.history,
    lead.objetivo_compra,
    lead.objective,
    lead.preferencia_bairro,
    lead.localizacao,
    lead.budget_max,
    lead.budget_min,
    lead.quartos,
    lead.prazo_compra,
    lead.renda_mensal,
  ].filter(Boolean).join(' '));
}

function inferredLead(body) {
  const lead = body?.lead && typeof body.lead === 'object' ? { ...body.lead } : {};
  const text = combinedConversationText(body);

  if (!lead.objetivo_compra && !lead.objective) {
    if (/\b(aluguel|alugar|locacao|locar|mudar|mudar se)\b/.test(text)) {
      lead.objetivo_compra = 'aluguel';
    } else if (/\b(compra|comprar|financiar|financiamento)\b/.test(text)) {
      lead.objetivo_compra = 'compra';
    } else if (/\b(venda|vender)\b/.test(text)) {
      lead.objetivo_compra = 'venda';
    }
  }

  if (!lead.localizacao && !lead.preferencia_bairro) {
    const regionMatch = text.match(/\b(?:em|bairro|regiao|regiao de|perto de)\s+([a-z0-9 ]{3,40})/);
    if (regionMatch?.[1]) lead.preferencia_bairro = regionMatch[1].trim();
  }

  if (!lead.budget_max && !lead.budget_min && /\b(?:r\s*)?\$?\s?\d{3,}(?:[.,]\d{2})?\b/.test(text)) {
    const valueMatch = text.match(/\b(?:r\s*)?\$?\s?(\d{3,})(?:[.,]\d{2})?\b/);
    if (valueMatch?.[1]) lead.budget_max = Number(valueMatch[1]);
  }

  if (!lead.quartos && /\b\d+\s*(?:ou\s*\d+\s*)?(?:quarto|quartos|dormitorio|dormitorios)\b/.test(text)) {
    const roomMatch = text.match(/\b(\d+\s*(?:ou\s*\d+)?)\s*(?:quarto|quartos|dormitorio|dormitorios)\b/);
    if (roomMatch?.[1]) lead.quartos = roomMatch[1];
  }

  if (!lead.prazo_compra && /\b(hoje|amanha|este mes|esse mes|15 dias|dia \d{1,2}|\d{1,2} de [a-z]+|sem pressa)\b/.test(text)) {
    const prazoMatch = text.match(/\b(hoje|amanha|este mes|esse mes|15 dias|dia \d{1,2}|\d{1,2} de [a-z]+|sem pressa)\b/);
    if (prazoMatch?.[1]) lead.prazo_compra = prazoMatch[1];
  }

  if (!lead.renda_mensal && /\b(clt|autonomo|autonomo|pj|mei|renda)\b/.test(text)) {
    lead.renda_mensal = lead.renda_mensal || 'informada no historico';
  }

  return lead;
}

function needsGuidedQualification(body) {
  const lead = inferredLead(body);

  return !(lead.objetivo_compra || lead.objective)
    || (!lead.localizacao && !lead.preferencia_bairro)
    || (!lead.budget_max && !lead.budget_min)
    || !lead.quartos
    || !lead.prazo_compra;
}

function enforceSingleQuestion(content) {
  const text = String(content || '').trim();
  const firstQuestion = text.indexOf('?');
  if (firstQuestion === -1) return text;

  const secondQuestion = text.indexOf('?', firstQuestion + 1);
  if (secondQuestion === -1) return text;

  return text.slice(0, firstQuestion + 1).trim();
}

function appearsToAddUnprovidedPropertyDetails(content, body) {
  const text = normalizeText(content);
  const propertyText = normalizeText(JSON.stringify(body?.properties || []));
  const riskyTerms = [
    'banheiro',
    'banheiros',
    'sala',
    'cozinha',
    'area util',
    'area de servico',
    'metro',
    'metros',
    'm2',
    'm²',
    'transporte',
    'publico',
    'bairro tranquilo',
    'comodidades',
    'supermercado',
    'supermercados',
    'farmacia',
    'farmacias',
    'restaurante',
    'restaurantes',
    'proximo',
    'proximos',
    'condominio',
    'lazer',
    'varanda',
    'piscina',
    'elevador',
  ];

  return riskyTerms.some((term) => text.includes(term) && !propertyText.includes(normalizeText(term)));
}

function appearsToInventPropertyWithoutContext(content) {
  const text = String(content || '');
  const normalized = normalizeText(text);

  return /(?:^|\n)\s*(?:1[.)]|1️⃣|2[.)]|2️⃣|3[.)]|3️⃣)|R\$\s*\d|op[cç][oõ]es?:/i.test(text)
    || /\b(?:casa|apartamento|cobertura|terreno)\s+\d+\b/i.test(text)
    || /\bcodigo\s+[a-z0-9-]+\b/i.test(text)
    || normalized.includes('casa 123')
    || normalized.includes('imovel que mais se encaixa')
    || normalized.includes('tenho este imovel')
    || normalized.includes('tenho estas opcoes');
}

function appearsToConfuseLeadObjective(content, body) {
  const text = normalizeText(content);
  const lead = inferredLead(body);
  const objective = normalizeText(lead.objetivo_compra || lead.objective || '');

  if (text.includes('exaple') || text.includes('example')) return true;

  if (objective === 'aluguel') {
    return text.includes('sua compra')
      || text.includes('para sua compra')
      || text.includes('comprar esse imovel')
      || text.includes('valor de compra')
      || text.includes('venda r');
  }

  if (objective === 'compra') {
    return text.includes('seu aluguel')
      || text.includes('para alugar')
      || text.includes('aluguel mensal')
      || text.includes('valor de aluguel');
  }

  return false;
}

function safeModelReply(content, body, model, provider) {
  if (!hasRealProperties(body) && appearsToInventPropertyWithoutContext(content)) {
    return {
      ...localTrainedReply(body),
      provider: 'local-trained',
      fallback: true,
      warnings: [`${provider} response discarded because it appeared to invent properties`],
    };
  }

  if (hasRealProperties(body) && appearsToAddUnprovidedPropertyDetails(content, body)) {
    const grounded = groundedPropertyReply(body) || localTrainedReply(body);
    return {
      ...grounded,
      provider: 'local-trained',
      fallback: true,
      warnings: [`${provider} response discarded because it added unprovided property details`],
    };
  }

  if (appearsToConfuseLeadObjective(content, body)) {
    const fallback = hasRealProperties(body)
      ? (groundedPropertyReply(body) || localTrainedReply(body))
      : localTrainedReply(body);

    return {
      ...fallback,
      provider: 'local-trained',
      fallback: true,
      warnings: [`${provider} response discarded because it confused the lead objective`],
    };
  }

  return { content: enforceSingleQuestion(content), model, provider };
}

function propertyLine(property) {
  const bedrooms = Number(property?.dormitorios || property?.bedrooms || 0) + Number(property?.suites || 0);
  const rent = Number(property?.valor_aluguel || property?.rent_price || property?.rent || 0);
  const sale = Number(property?.valor_venda || property?.sale_price || property?.price || 0);
  const price = rent > 0
    ? `aluguel R$ ${rent.toLocaleString('pt-BR')}`
    : sale > 0
      ? `venda R$ ${sale.toLocaleString('pt-BR')}`
      : 'valor sob consulta';

  return [
    `Codigo ${property?.codigo_imovel || property?.codigo || property?.code || 'N/A'}`,
    property?.tipo_imovel || property?.type || 'Imovel',
    [property?.bairro || property?.neighborhood, property?.cidade || property?.city].filter(Boolean).join(', ') || 'local nao informado',
    `${bedrooms || '?'} quartos`,
    price,
  ].join(' | ');
}

function groundedPropertyReply(body) {
  const properties = Array.isArray(body.properties) ? body.properties.slice(0, 3) : [];
  if (!properties.length) return null;

  const lines = properties.map((property) => `- ${propertyLine(property)}`);
  const plural = properties.length > 1;

  return {
    content: [
      plural
        ? 'Tenho estas opcoes reais dentro do contexto do atendimento:'
        : 'Tenho este imovel real dentro do contexto do atendimento:',
      lines.join('\n'),
      'Posso chamar um corretor para confirmar os detalhes que nao aparecem aqui e seguir com voce?',
    ].join('\n\n'),
    model: 'local-grounded-properties',
    provider: 'local-trained',
  };
}

function buildPrompt(body) {
  const assistantName = body.assistant_name || 'Teresa';
  const companyName = body.company_name || 'Exclusiva Lar Imoveis';
  const properties = Array.isArray(body.properties) ? body.properties.slice(0, MAX_PROPERTIES) : [];
  const lead = inferredLead(body);
  const missing = [];

  if (!lead.objetivo_compra && !lead.objective) missing.push('compra ou aluguel');
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
- Nao use lista de perguntas. Se faltarem varios dados, escolha somente o dado mais importante agora.
- Nao invente imoveis: use somente os imoveis reais abaixo.
- Nao invente caracteristicas de imoveis. Nao cite metragem, banheiro, sala, cozinha, transporte, condominio, varanda, lazer ou qualquer detalhe que nao esteja explicitamente nos imoveis reais.
- Se o cliente pedir detalhes que nao constam nos imoveis reais, diga com educacao que vai confirmar esses detalhes com o corretor.
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

  return { content: enforceSingleQuestion(content), model };
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

  return safeModelReply(content, body, model, 'pollinations');
}

async function huggingFaceChat(body) {
  if (!HF_TOKEN) {
    throw new Error('HF_TOKEN not configured');
  }

  const prompt = buildPrompt(body);
  const model = body.hf_model || HF_MODEL;

  const response = await fetch(`${HF_ROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      stream: false,
      temperature: 0.35,
      max_tokens: Number(body.max_tokens || 260),
      messages: [
        { role: 'system', content: prompt.system },
        { role: 'user', content: prompt.user },
      ],
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HuggingFace ${response.status}: ${text.slice(0, 500)}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('Resposta invalida do Hugging Face');
  }

  return safeModelReply(content, body, model, 'huggingface');
}

async function localModelChat(body) {
  if (!LOCAL_MODEL_BASE_URL) {
    throw new Error('LOCAL_MODEL_BASE_URL not configured');
  }

  const prompt = buildPrompt(body);
  const headers = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
  };
  if (LOCAL_MODEL_API_KEY) headers.Authorization = `Bearer ${LOCAL_MODEL_API_KEY}`;

  const response = await fetch(`${LOCAL_MODEL_BASE_URL}/chat`, {
    method: 'POST',
    headers,
    signal: AbortSignal.timeout(Math.max(5000, LOCAL_MODEL_TIMEOUT_MS)),
    body: JSON.stringify({
      message: prompt.user,
      history: prompt.system,
      assistant_name: body.assistant_name || 'Teresa',
      company_name: body.company_name || 'Exclusiva Lar Imoveis',
      max_new_tokens: Math.max(
        48,
        Math.min(Number(body.max_new_tokens || body.max_tokens || 90), 120),
      ),
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`LocalModel ${response.status}: ${text.slice(0, 500)}`);
  }

  const data = await response.json();
  const content = data?.content;
  if (!content || typeof content !== 'string') {
    throw new Error('Resposta invalida do modelo local fine-tuned');
  }

  return safeModelReply(content, body, data?.model || LOCAL_MODEL_NAME, 'local-finetuned');
}

function localTrainedReply(body) {
  const lead = inferredLead(body);
  const message = String(body.message || '').trim();
  const name = lead.nome || lead.name || lead.cliente || '';
  const firstName = String(name).split(/\s+/).filter(Boolean)[0] || '';
  const prefix = firstName ? `${firstName}, ` : '';

  const examples = selectTrainingExamples(body);
  const best = examples[0];
  const lower = normalizeText(message);
  const grounded = groundedPropertyReply(body);

  if (grounded && /\b(gostei|interesse|quero esse|quero este|esse|este|codigo|cod|ref|referencia|visita|ver)\b/.test(lower)) {
    return grounded;
  }

  if (/\b(oi|ola|bom dia|boa tarde|boa noite)\b/.test(lower)) {
    return {
      content: `${prefix}que bom falar com voce. Me conte, por favor: voce procura comprar, alugar ou vender um imovel? Exemplo: "quero alugar apartamento ate R$ 1.500".`,
      model: 'local-trained-rules',
      provider: 'local-trained',
    };
  }

  if (!lead.objetivo_compra && !lead.objective) {
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

  if (grounded) return grounded;

  if (!hasRealProperties(body)) {
    return {
      content: `${prefix}perfeito, obrigado pelas informacoes. Vou usar esses criterios para buscar somente opcoes reais compativeis. Se voce quiser, posso tambem chamar um corretor para continuar com voce.`,
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

  if (!hasRealProperties(body) && needsGuidedQualification(body)) {
    return localTrainedReply(body);
  }

  if (AI_PROVIDER === 'ollama') {
    const result = await ollamaChat(body);
    return { ...result, provider: 'ollama' };
  }

  if (AI_PROVIDER === 'pollinations') {
    return pollinationsChat(body);
  }

  if (AI_PROVIDER === 'huggingface' || AI_PROVIDER === 'hf') {
    return huggingFaceChat(body);
  }

  if (AI_PROVIDER === 'local_model' || AI_PROVIDER === 'finetuned') {
    return localModelChat(body);
  }

  if (AI_PROVIDER === 'local') {
    return localTrainedReply(body);
  }

  if (LOCAL_MODEL_BASE_URL) {
    try {
      return await localModelChat(body);
    } catch (error) {
      errors.push(`local_model: ${error.message}`);
    }
  }

  try {
    return await huggingFaceChat(body);
  } catch (error) {
    errors.push(`huggingface: ${error.message}`);
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

  if (AI_PROVIDER === 'huggingface' || AI_PROVIDER === 'hf') {
    return res.json({
      success: Boolean(HF_TOKEN),
      provider: 'huggingface',
      model: HF_MODEL,
      configured: Boolean(HF_TOKEN),
      training: trainingHealth,
      error: HF_TOKEN ? null : 'HF_TOKEN not configured',
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

  if (AI_PROVIDER === 'local_model' || AI_PROVIDER === 'finetuned') {
    return res.json({
      success: Boolean(LOCAL_MODEL_BASE_URL),
      provider: 'local-finetuned',
      model: LOCAL_MODEL_NAME,
      configured: Boolean(LOCAL_MODEL_BASE_URL),
      base_url: LOCAL_MODEL_BASE_URL || null,
      training: trainingHealth,
      error: LOCAL_MODEL_BASE_URL ? null : 'LOCAL_MODEL_BASE_URL not configured',
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
        huggingface_fallback: Boolean(HF_TOKEN),
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
