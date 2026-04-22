import 'dotenv/config'

import { Document } from '@langchain/core/documents'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { OpenAIEmbeddings } from '@langchain/openai'

const movies = [
  {
    id: 1,
    title: 'Stepbrother',
    description: `Comedic journey full of adult humor and awkwardness.`
  },
  {
    id: 2,
    title: 'The Matrix',
    description: `Deals with alternate realities and questioning what's real.`
  },
  {
    id: 3,
    title: 'Shutter Island',
    description: `A mind-bending plot with twists and turns.`
  },
  {
    id: 4,
    title: 'Memento',
    description: `A non-linear narrative that challenges the viewer's perception.`
  },
  {
    id: 5,
    title: 'Doctor Strange',
    description: `Features alternate dimensions and reality manipulation.`
  },
  {
    id: 6,
    title: 'Paw Patrol',
    description: `Children's animated movie where a group of adorable puppies save people from all sorts of emergencies.`
  },
  {
    id: 7,
    title: 'Interstellar',
    description: `Features futuristic space travel with high stakes`
  }
]

const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-large',
  apiKey: process.env.AI_API_KEY,
  configuration: {
    baseURL: process.env.AI_BASE_URL
  }
})

const vectorStore = () =>
  MemoryVectorStore.fromDocuments(
    movies.map(
      (movie) =>
        new Document({
          pageContent: `Title: ${movie.title}\n${movie.description}`, // non-structured
          metadata: { source: movie.id, title: movie.title } // structured
        })
    ),
    embeddings
  )

const search = async (query, count = 2) => {
  const store = await vectorStore()
  return store.similaritySearchWithScore(query, count)
}

console.log(await search('a movie that will make me feel like crazy'))
