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
// 🧠 tools
// =========================
function calculateAverage(data) {
  const sum = data.reduce((acc, item) => acc + item.sales, 0);
  return sum / data.length;
}

function getMax(data) {
  return Math.max(...data.map((i) => i.sales));
}

// =========================
// 🚀 analyze Agent
// =========================
app.post("/analyze", async (req, res) => {
  const { data } = req.body;

  // 🧠 查找历史记忆
  const memory = await searchMemory(JSON.stringify(data));

  let messages = [
    {
      role: "system",
      content: `
你是一个数据分析Agent。

规则：
1. 必须参考历史经验（如果有）
2. 涉及计算必须调用工具
3. 不要编造数据
4. 最终输出必须包含“最终结论”

历史经验：
${memory ? memory.text : "暂无历史经验"}
`,
    },
    {
      role: "user",
      content: `分析以下数据：${JSON.stringify(data)}`,
    },
  ];

  try {
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
      // 🧠 final answer
      // =========================
      if (message.content?.includes("最终结论")) {
        // 🧠 存入 memory
        await saveMemory(message.content);

        return res.json({
          result: message.content,
        });
      }
    }

    res.json({
      result: "未在限定轮次内完成任务",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// =========================
// 🚀 start server
// =========================
app.listen(3000, () => {
  console.log("🚀 Hermes Agent running on http://localhost:3000");
});