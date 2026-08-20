import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const backendResponse = await fetch(`${BACKEND_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      return NextResponse.json(
        { error: `Backend error: ${backendResponse.status}`, details: errorText },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to reach backend', details: String(error) },
      { status: 502 }
    )
  }
}
