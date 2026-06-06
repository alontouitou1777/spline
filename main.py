"""
תוכנית ראשית — אינטרפולציה ספליין קובי
========================================
מגישים: אלון טואיטו וסופי טננבאום

מגדירה טבלת נקודות לדוגמה, קוראת לפונקציית האינטרפולציה,
ומדפיסה את התוצאה עם הסבר מלא.
"""

import numpy as np


def cubic_spline(xs: list[float], ys: list[float], x_query: float) -> float:
    """
    מחשבת אינטרפולציה ספליין קובי בנקודה x_query.

    פרמטרים:
        xs      - רשימת ערכי איקס, חייבת להיות ממוינת מקטן לגדול
        ys      - רשימת ערכי וואי המתאימים לכל איקס
        x_query - הנקודה שרוצים למצוא את ערכה

    מחזירה:
        ערך משוערך של העקומה בנקודה x_query
    """

    n = len(xs)

    # בדיקות קלט - מוודאים שהנתונים תקינים לפני כל חישוב
    if len(ys) != n:
        raise ValueError("רשימות האיקס והוואי חייבות להיות באותו אורך")
    if n < 2:
        raise ValueError("צריך לפחות שתי נקודות לאינטרפולציה")
    if not (xs[0] <= x_query <= xs[-1]):
        raise ValueError(f"הנקודה {x_query} מחוץ לתחום [{xs[0]}, {xs[-1]}]")

    # ממירים למערכי numpy כדי שנוכל לעבוד עליהם בפעולות מטריצה
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # ── צעד 1: חישוב הרווחים בין נקודות סמוכות ──────────────────
    # h[i] = המרחק האופקי בין xs[i] ל-xs[i+1]
    h = np.diff(xs)

    # ── צעד 2: בניית מטריצת המקדמים A ווקטור הצד הימני b ────────
    # המטריצה תלת-אלכסונית: כל שורה פנימית i מייצגת את תנאי
    # החלקות בנקודה i. שורות הקצה מייצגות תנאי Natural Spline
    # (עיקול אפס בשני הקצוות).
    A = np.zeros((n, n))
    A[0, 0] = 1.0        # תנאי קצה שמאלי: c[0] = 0
    A[n - 1, n - 1] = 1.0  # תנאי קצה ימני: c[n-1] = 0

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]              # אלכסון תחתון
        A[i, i]     = 2 * (h[i - 1] + h[i]) # אלכסון ראשי
        A[i, i + 1] = h[i]                   # אלכסון עליון

    # בניית וקטור הצד הימני: שיעור השינוי בשיפוע בכל נקודה פנימית
    b = np.zeros(n)
    for i in range(1, n - 1):
        slope_right = (ys[i + 1] - ys[i]) / h[i]
        slope_left  = (ys[i] - ys[i - 1]) / h[i - 1]
        b[i] = 3 * (slope_right - slope_left)

    # ── צעד 3: פתרון מערכת המשוואות A*c = b ─────────────────────
    # התוצאה היא וקטור העיקולים c - הנגזרת השנייה בכל נקודה
    c = np.linalg.solve(A, b)

    # ── צעד 4: חישוב שאר המקדמים לכל קטע ───────────────────────
    # הפולינום בקטע i: S(x) = a + b_c*t + c*t^2 + d*t^3
    # כאשר t = x - xs[i]
    a   = ys[:-1]
    d   = (c[1:] - c[:-1]) / (3 * h)
    b_c = (ys[1:] - ys[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3

    # ── צעד 5: מציאת הקטע הנכון וחישוב הערך ─────────────────────
    # searchsorted מוצא את האינדקס i כך ש- xs[i] <= x_query < xs[i+1]
    i = int(np.clip(np.searchsorted(xs, x_query, side='right') - 1, 0, n - 2))
    t = x_query - xs[i]

    return float(a[i] + b_c[i] * t + c[i] * t**2 + d[i] * t**3)


def print_table(hours: list, temperatures: list) -> None:
    """מדפיסה את טבלת הנקודות הידועות בצורה מסודרת."""
    print("  +-------------+------------------+")
    print("  |    שעה      |   טמפרטורה       |")
    print("  +-------------+------------------+")
    for hour, temp in zip(hours, temperatures):
        print(f"  |  {hour:02d}:00       |   {temp:>5.1f} מעלות    |")
    print("  +-------------+------------------+")


def find_neighbors(hours: list, temperatures: list, target: float):
    """מחזירה את הנקודות הסמוכות לנקודת החיזוי."""
    hour_before = None
    temp_before = None
    hour_after  = None
    temp_after  = None

    for i in range(len(hours) - 1):
        if hours[i] <= target <= hours[i + 1]:
            hour_before = hours[i]
            temp_before = temperatures[i]
            hour_after  = hours[i + 1]
            temp_after  = temperatures[i + 1]
            break

    return hour_before, temp_before, hour_after, temp_after


def main() -> None:

    # ── הגדרת טבלת הנקודות הידועות ───────────────────────────────
    # מדידות טמפרטורה בתחנת מזג אוויר לאורך יום אחד
    hours        = [0,  3,  6,  9,  12, 15, 18, 21]
    temperatures = [14, 11, 9,  13, 24, 28, 23, 17]

    # הנקודה שרוצים לאמוד - שעה שלא נמדדה ישירות
    target_hour = 10.5

    # ── הדפסת כותרת ומידע על הנקודות הידועות ────────────────────
    print()
    print("+==================================================+")
    print("|        אינטרפולציה ספליין קובי - דוגמה          |")
    print("+==================================================+")
    print()
    print("נקודות המדידה הידועות:")
    print()
    print_table(hours, temperatures)

    # ── הצגת השאלה ───────────────────────────────────────────────
    print()
    print(f"שאלה: מה הייתה הטמפרטורה בשעה {target_hour:.1f}?")
    print()
    print("   שעה זו לא נמדדה ישירות - נשתמש באינטרפולציה ספליין")
    print("   כדי לאמוד אותה מתוך הנקודות הסמוכות.")

    # ── קריאה לפונקציית האינטרפולציה ────────────────────────────
    print()
    print("מחשב...")
    print()

    result = cubic_spline(hours, temperatures, target_hour)

    # ── הדפסת התוצאה עם הקשר ─────────────────────────────────────
    hour_before, temp_before, hour_after, temp_after = find_neighbors(
        hours, temperatures, target_hour
    )

    print("+==================================================+")
    print("|                   תוצאה                         |")
    print("+==================================================+")
    if hour_before is not None:
        print(f"|  שעה {hour_before:02d}:00  ->  {temp_before:.1f} מעלות  (מדידה ידועה)       |")
    print(f"|  שעה {target_hour:.2f} ->  {result:.2f} מעלות  <- תוצאת האינטרפולציה  |")
    if hour_after is not None:
        print(f"|  שעה {hour_after:02d}:00  ->  {temp_after:.1f} מעלות  (מדידה ידועה)       |")
    print("+==================================================+")
    print()
    print(f"הטמפרטורה המשוערת בשעה {target_hour:.1f} היא: {result:.2f} מעלות צלזיוס.")
    print()
    print("   הערך הגיוני - הוא נמצא בין הנקודות הסמוכות")
    print("   ומשקף את העקומה החלקה שעוברת דרך כל המדידות.")
    print()


if __name__ == "__main__":
    main()

