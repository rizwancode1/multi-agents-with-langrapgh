import { NextResponse } from 'next/server'
import { BACKEND_URL, backendHeaders } from '../../../lib/backend'

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/conversations`, {
      cache: 'no-store',
      headers: backendHeaders(),
    })
    if (!res.ok) {
      return NextResponse.json({ error: 'Backend error', status: res.status }, { status: res.status })
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend', details: String(error) }, { status: 502 })
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const res = await fetch(`${BACKEND_URL}/conversations`, {
      method: 'POST',
      headers: backendHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const text = await res.text()
      return NextResponse.json({ error: 'Backend error', details: text }, { status: res.status })
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to reach backend', details: String(error) }, { status: 502 })
  }
}
