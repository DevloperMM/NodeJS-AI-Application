import math from 'advanced-calculator'
import openai from './openai.js'

const QUESTION = process.argv[2] || 'evaluate 2+3'

const messages = [
  {
    role: 'system',
    content:
      'You are a math calculator assistant. Solve math problems accurately and briefly. Show short steps only when needed. Always give a clear final answer in plain text mentioning what you answered for what. If the question is unclear, ask one short clarifying question'
  },
  {
    role: 'user',
    content: QUESTION
  }
]

const functions = {
  calculate({ expression }) {
    return math.evaluate(expression)
  }
}

const getCompletion = (messages) => {
  return openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages,
    temperature: 0,
    tool_choice: 'auto',
    tools: [
      {
        type: 'function',
        function: {
          name: 'calculate',
          description: 'Run in cases of evaluating math expressions',
          parameters: {
            type: 'object',
            properties: {
              expression: {
                type: 'string',
                description:
                  'The math expressions to evaluate to evaluate like "2*4+(21/3)"'
              }
            },
            required: ['expression']
          }
        }
      }
    ]
  })
}

while (true) {
  const response = await getCompletion(messages)
  // console.log(JSON.stringify(response, null, 2))

  const msg = response.choices[0].message

  if (!msg.tool_calls || msg.tool_calls.length === 0) {
    console.log(msg.content)

    // Final messages array to push back to model
    messages.push({
      role: msg.role,
      content: msg.content
    })
    console.log(JSON.stringify(messages, null, 2))

    break
  }

  messages.push({
    role: 'assistant',
    tool_calls: msg.tool_calls
  })

  for (const toolCall of msg.tool_calls) {
    const fnName = toolCall.function.name
    const args = JSON.parse(toolCall.function.arguments)
    const result = functions[fnName](args)
    messages.push({
      role: 'tool',
      tool_call_id: toolCall.id,
      content: JSON.stringify({ result })
    })
  }
}
