/** 34 类配色：黄金角散列，保证相邻 class_id 颜色差异够大。 */

const GOLDEN = 137.508

export function classHue(classId: number): number {
  return (classId * GOLDEN) % 360
}

export function classStroke(classId: number, alpha = 1): string {
  return `hsla(${classHue(classId)}, 78%, 62%, ${alpha})`
}

export function classFill(classId: number, alpha = 0.22): string {
  return `hsla(${classHue(classId)}, 78%, 58%, ${alpha})`
}

/** 给 HTML 元素用的实色（队列小圆点等） */
export function classSolid(classId: number): string {
  return `hsl(${classHue(classId)}, 78%, 62%)`
}
