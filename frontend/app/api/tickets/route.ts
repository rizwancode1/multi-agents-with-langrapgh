import { NextResponse } from 'next/server'
import { BACKEND_URL, backendHeaders } from '../../../lib/backend'

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const qs = searchParams.toString()
    const res = await fetch(`${BACKEND_URL}/tickets${qs ? `?${qs}` : ''}`, {
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
