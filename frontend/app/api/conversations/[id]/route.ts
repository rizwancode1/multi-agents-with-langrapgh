import { NextResponse } from 'next/server'
import { BACKEND_URL, backendHeaders } from '../../../../lib/backend'

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  try {
    const res = await fetch(`${BACKEND_URL}/conversations/${id}`, {
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
