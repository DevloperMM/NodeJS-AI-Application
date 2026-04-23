// run `node qa.js ${question here in quotes} directly`

import 'dotenv/config'

import { Document } from '@langchain/core/documents'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { OpenAIEmbeddings } from '@langchain/openai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { existsSync } from 'node:fs'
import fs from 'node:fs/promises'
import openai from './openai.js'

const question = process.argv[2] || 'who is an women enterpreneur?'
const video = 'https://youtu.be/C842vFY5kRo?si=NMSGecf1AF7MxcuE'

const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-large',
  apiKey: process.env.AI_API_KEY,
  configuration: {
    baseURL: process.env.AI_BASE_URL
  }
})

const vectorStore = (docs) => MemoryVectorStore.fromDocuments(docs, embeddings)

function getYouTubeId(url) {
  const regex =
    /(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/
  const match = url.match(regex)
  return match ? match[1] : null
}

const docsFromYTVideo = async (video) => {
  try {
    const videoID = getYouTubeId(video)
    const dir = 'transcripts'
    const filePath = `./${dir}/${videoID}.txt`
    const fileJSON = `./${dir}/${videoID}.json`

    await fs.mkdir(dir, { recursive: true })

    let script = ''

    if (existsSync(filePath)) {
      console.log('Loading video transcript...')
      script = await fs.readFile(filePath, 'utf-8')
    } else {
      console.log('Fetching from API...')

      const response = await fetch(
        'https://www.youtube-transcript.io/api/transcripts',
        {
          method: 'POST',
          headers: {
            Authorization: `Basic ${process.env.TRANSCRIPT_API}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ ids: [videoID] })
        }
      )

      const data = await response.json()

      await fs.writeFile(fileJSON, JSON.stringify(data, null, 2))

      const scriptArr = data[0]?.tracks[0]?.transcript || []
      script = scriptArr.map((item) => item.text).join(' ')

      if (script.trim()) {
        await fs.writeFile(filePath, script)
        console.log('Transcript saved locally !!')
      } else {
        console.log('API returned empty transcript')
        return []
      }
    }

    const videoDocs = new Document({
      pageContent: script,
      metadata: {
        source: video,
        title: video.title,
        ids: [videoID]
      }
    })

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2500,
      chunkOverlap: 100
    })

    return splitter.splitDocuments([videoDocs])
  } catch (error) {
    console.error(error)
    return []
  }
}

const docsFromPdf = async () => {
  try {
    console.log('Loading PDF...')
    const loader = new PDFLoader('./women.pdf')
    const docs = await loader.load()
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2500,
      chunkOverlap: 200
    })

    return splitter.splitDocuments(docs)
  } catch (error) {
    console.error(error)
    return []
  }
}

const loadStore = async () => {
  const videoDocs = await docsFromYTVideo(video)
  const pdfDocs = await docsFromPdf()

  // console.log(pdfDocs[0], videoDocs[0])

  return vectorStore([...pdfDocs, ...videoDocs], embeddings)
}

const query = async () => {
  const store = await loadStore()
  const results = await store.similaritySearch(question, 2)

  // console.log(results)

  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    temperature: 0,
    messages: [
      {
        role: 'assistant',
        content:
          'You are a helpful AI assistant. Answser questions to your best ability.'
      },
      {
        role: 'user',
        content: `Answer the following question using the provided context. If you cannot answer the question with the context, don't lie and make up stuff. Just say you need more context.
        Question: ${question}\n
        Context: ${results.map((r) => r.pageContent).join('\n')}`
      }
    ]
  })

  console.log(
    `Answer: ${response.choices[0].message.content}\n\nSources: ${results.map((r) => r.metadata.source).join(', ')}`
  )
}

query()
