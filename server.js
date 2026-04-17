import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.ARK_API_KEY,
  baseURL: process.env.ARK_BASE_URL
});

// ⭐ 核心：一个“带角色的Agent”
app.post("/analyze", async (req, res) => {
  const { data } = req.body;

  let messages = [
    {
      role: "system",
      content: `
你是一个数据分析Agent：

规则：
1. 分步骤完成任务
2. 需要计算必须调用工具
3. 完成后输出“最终结论”
`,
    },
    {
      role: "user",
      content: `分析数据：${JSON.stringify(data)}`,
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

      // ✅ 打印 Agent 思考过程
      console.log("🧠 当前 messages：");
      console.log(JSON.stringify(messages, null, 2));

      // 👉 如果 AI 调用工具
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

        // 把工具结果喂回去
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify({ result }),
        });

        continue; // 👉 继续循环
      }

      // 👉 如果AI给出最终答案
      if (message.content.includes("最终结论")) {
        return res.json({
          result: message.content,
        });
      }
    }

    res.json({
      result: "未能在限制步数内完成任务",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});

function calculateAverage(data) {
  const sum = data.reduce((acc, item) => acc + item.sales, 0);
  return sum / data.length;
}

function getMax(data) {
  return Math.max(...data.map(i => i.sales));
}