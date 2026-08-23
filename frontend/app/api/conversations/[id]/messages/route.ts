import { NextResponse } from 'next/server'
import { BACKEND_URL, backendHeaders } from '../../../../../lib/backend'

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  try {
    const body = await request.json()
    const res = await fetch(`${BACKEND_URL}/conversations/${id}/messages`, {
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
