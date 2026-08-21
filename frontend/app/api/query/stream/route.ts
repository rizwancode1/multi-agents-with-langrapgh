import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

export async function POST(request: Request) {
  try {
    const body = await request.text()
    const backendResponse = await fetch(`${BACKEND_URL}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body,
    })

    if (!backendResponse.ok) {
      return new NextResponse(
        await backendResponse.text(),
        { status: backendResponse.status }
      )
    }

    return new Response(backendResponse.body, {
      status: backendResponse.status,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
    })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to reach backend', details: String(error) },
      { status: 502 }
    )
  }
}
