import { NextResponse } from 'next/server'
import { BACKEND_URL, backendHeaders } from '../../../../lib/backend'

type Params = { params: Promise<{ id: string }> }

export async function GET(_request: Request, { params }: Params) {
  try {
    const { id } = await params
    const res = await fetch(`${BACKEND_URL}/tickets/${encodeURIComponent(id)}`, {
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

export async function PATCH(request: Request, { params }: Params) {
  try {
    const { id } = await params
    const body = await request.json()
    const res = await fetch(`${BACKEND_URL}/tickets/${encodeURIComponent(id)}`, {
      method: 'PATCH',
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
