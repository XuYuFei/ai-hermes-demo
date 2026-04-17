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

  try {
    const completion = await client.chat.completions.create({
      model: process.env.ARK_MODEL,
      messages: [
        {
          role: "system",
          content: `
你是一个专业数据分析师，请完成以下任务：
1. 分析数据结构
2. 找出关键指标
3. 给出结论
4. 用简单易懂的语言说明
`,
        },
        {
          role: "user",
          content: `请分析以下数据：${JSON.stringify(data)}`,
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "calculateAverage",
            description: "计算销售数据的平均值",
            parameters: {
              type: "object",
              properties: {
                data: {
                  type: "array",
                  items: {
                    type: "object",
                    properties: {
                      sales: { type: "number" }
                    }
                  }
                }
              },
              required: ["data"]
            }
          }
        }
      ]
    });

    const message = completion.choices[0].message;

    // 👉 如果 AI 要调用函数
    if (message.tool_calls) {
      const toolCall = message.tool_calls[0];
      const args = JSON.parse(toolCall.function.arguments);

      let result;

      if (toolCall.function.name === "calculateAverage") {
        result = calculateAverage(args.data);
      }

      // 👉 把结果再喂给 AI
      const secondResponse = await client.chat.completions.create({
        model: process.env.ARK_MODEL,
        messages: [
          ...messages,
          message,
          {
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify({ result }),
          }
        ]
      });

      return res.json({
        result: secondResponse.choices[0].message.content
      });
    }

    res.json({
      result: completion.choices[0].message.content,
    });
  } catch (err) {
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