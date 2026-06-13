import { useEffect, useRef, useState } from 'react'
import type { Detection } from '../../../shared/api-types'

// X-ray rasm + bounding box overlay. bbox ASL rasm pikselida ([x1,y1,x2,y2]);
// ko'rsatilayotgan o'lchamga proporsional masshtablanadi.
export function ImageWithDetections({
  src,
  detections,
  naturalSize
}: {
  src: string
  detections: Detection[]
  // Agar backend asl o'lchamni bermasa, rasm yuklanganda o'lchaymiz.
  naturalSize?: { w: number; h: number }
}): JSX.Element {
  const imgRef = useRef<HTMLImageElement>(null)
  const [nat, setNat] = useState<{ w: number; h: number } | null>(naturalSize ?? null)
  const [disp, setDisp] = useState<{ w: number; h: number } | null>(null)

  useEffect(() => {
    const el = imgRef.current
    if (!el) return
    const measure = (): void => {
      setDisp({ w: el.clientWidth, h: el.clientHeight })
      if (!nat && el.naturalWidth) setNat({ w: el.naturalWidth, h: el.naturalHeight })
    }
    if (el.complete) measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [nat])

  const sx = nat && disp ? disp.w / nat.w : 1
  const sy = nat && disp ? disp.h / nat.h : 1

  return (
    <div className="imgwrap">
      <img
        ref={imgRef}
        src={src}
        alt="X-ray skaner"
        onLoad={(e) => {
          const el = e.currentTarget
          if (!naturalSize) setNat({ w: el.naturalWidth, h: el.naturalHeight })
          setDisp({ w: el.clientWidth, h: el.clientHeight })
        }}
      />
      {disp &&
        detections.map((d, i) => {
          const [x1, y1, x2, y2] = d.bbox
          const style = {
            left: `${x1 * sx}px`,
            top: `${y1 * sy}px`,
            width: `${(x2 - x1) * sx}px`,
            height: `${(y2 - y1) * sy}px`
          }
          return (
            <div className="bbox" key={i} style={style}>
              <span className="tag">
                {d.class} · {(d.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )
        })}
    </div>
  )
}
