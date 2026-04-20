import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// =========================
// 🧠 字节模型客户端
// =========================
const client = new OpenAI({
  apiKey: process.env.ARK_API_KEY,
  baseURL: process.env.ARK_BASE_URL,
});

// =========================
// 🧠 向量记忆库
// =========================
const memoryStore = [];

// =========================
// 🧠 embedding
// =========================
async function getEmbedding(text) {
  const res = await client.embeddings.create({
    model: process.env.ARK_EMBEDDING_MODEL,
    input: text
  })

  return res.data[0].embedding;
}

// =========================
// 🧠 cosine similarity
// =========================
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// =========================
// 🧠 保存记忆
// =========================
async function saveMemory(text) {
  const embedding = await getEmbedding(text);

  memoryStore.push({
    text,
    embedding,
  });
}

// =========================
// 🧠 检索记忆
// =========================
async function searchMemory(query) {
  if (memoryStore.length === 0) return null;

  const queryEmbedding = await getEmbedding(query);

  let best = null;
  let bestScore = -1;

  for (const mem of memoryStore) {
    const score = cosineSimilarity(queryEmbedding, mem.embedding);

    if (score > bestScore) {
      bestScore = score;
      best = mem;
    }
  }

  return best;
}

// =========================
// 🧠 Tools
// =========================
function calculateAverage(data) {
  const sum = data.reduce((acc, i) => acc + i.sales, 0);
  return sum / data.length;
}

function getMax(data) {
  return Math.max(...data.map(i => i.sales));
}

// =========================
// 🧠 Planner（新增核心）
// =========================
async function createPlan(data) {
  const res = await client.chat.completions.create({
    model: process.env.ARK_MODEL,
    messages: [
      {
        role: "system",
        content: `
你是一个数据分析规划专家。

请生成执行计划：
要求：
1. 按 step1, step2... 输出
2. 每一步要具体
3. 不要执行
`,
      },
      {
        role: "user",
        content: `分析数据：${JSON.stringify(data)}`,
      },
    ],
  });

  return res.choices[0].message.content;
}

// =========================
// 🚀 Agent 主流程
// =========================
app.post("/analyze", async (req, res) => {
  const { data } = req.body;

  try {
    // 🧠 1. 生成计划
    const plan = await createPlan(data);
    console.log('📄 Plan:\n', plan);

    // 🧠 2. 检索记忆
    const memory = await searchMemory(JSON.stringify(data));

    let messages = [
      {
        role: "system",
        content: `
你是一个严格执行计划的数据分析Agent。

执行规则：
1. 必须严格按照步骤执行
2. 每一步必须说明“正在执行 stepX”
3. 涉及计算必须调用工具
4. 不要重复步骤
5. 最终必须输出“最终结论”

执行计划：
${plan}

历史经验：
${memory ? memory.text : "无"}
`,
      },
      {
        role: "user",
        content: `分析数据：${JSON.stringify(data)}`,
      },
    ];

    let loopCount = 0;

    while (loopCount < 5) {
      loopCount++;

      const completion = await client.chat.completions.create({
        model: process.env.ARK_MODEL,
        messages,
        tools: [
          {
            type: "function",
            function: {
              name: "calculateAverage",
              description: "计算平均值",
              parameters: {
                type: "object",
                properties: {
                  data: { type: "array" },
                },
                required: ["data"],
              },
            },
          },
          {
            type: "function",
            function: {
              name: "getMax",
              description: "获取最大值",
              parameters: {
                type: "object",
                properties: {
                  data: { type: "array" },
                },
                required: ["data"],
              },
            },
          },
        ],
      });

      const message = completion.choices[0].message;
      messages.push(message);

      console.log(`\n🚀 第 ${loopCount} 轮`);
      console.log(JSON.stringify(message, null, 2));

      // =========================
      // 🧠 tool call
      // =========================
      if (message.tool_calls) {
        const toolCall = message.tool_calls[0];
        const args = JSON.parse(toolCall.function.arguments);

        let result;

        if (toolCall.function.name === "calculateAverage") {
          result = calculateAverage(args.data);
        }

        if (toolCall.function.name === "getMax") {
          result = getMax(args.data);
        }

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({ result }),
        });

        continue;
      }

      // =========================
      // ✅ 结束条件
      // =========================
      if (message.content?.includes("最终结论")) {
        // 🧠 存入 memory
        await saveMemory(message.content);

        return res.json({
          result: message.content,
          plan,
        });
      }
    }

    res.json({
      result: "未完成任务",
      plan,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});


// =========================
// 🚀 test Planner
// =========================
app.post("/plan", async (req, res) => {
  const { data } = req.body;

  try {
    const result = await createPlan(data);
    res.json({ result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// =========================
// 🚀 启动
// =========================
app.listen(3000, () => {
  console.log("🚀 Day6 Agent running http://localhost:3000");
});