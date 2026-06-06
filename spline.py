import numpy as np


def cubic_spline(xs, ys, x_query):
    n = len(xs)

    if len(ys) != n:
        raise ValueError("רשימות האיקס והוואי חייבות להיות באותו אורך")
    if n < 2:
        raise ValueError("צריך לפחות שתי נקודות")
    if not (xs[0] <= x_query <= xs[-1]):
        raise ValueError(f"הנקודה {x_query} מחוץ לתחום")

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # צעד 1: רווחים בין נקודות
    h = np.diff(xs)

    # צעד 2: מטריצה תלת-אלכסונית
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i]     = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    # צעד 2 המשך: וקטור הצד הימני
    b = np.zeros(n)
    for i in range(1, n - 1):
        slope_right = (ys[i + 1] - ys[i]) / h[i]
        slope_left  = (ys[i] - ys[i - 1]) / h[i - 1]
        b[i] = 3 * (slope_right - slope_left)

    # צעד 3: פתרון המערכת - וקטור העיקולים
    c = np.linalg.solve(A, b)

    # צעד 4: מקדמים לכל קטע
    a   = ys[:-1]
    d   = (c[1:] - c[:-1]) / (3 * h)
    b_c = (ys[1:] - ys[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3

    # צעד 5: מציאת הקטע וחישוב הערך
    idx = int(np.clip(np.searchsorted(xs, x_query, side='right') - 1, 0, n - 2))
    t   = x_query - xs[idx]

    return float(a[idx] + b_c[idx] * t + c[idx] * t**2 + d[idx] * t**3)


def main():
    # טבלת הנקודות הידועות
    hours        = [0,  3,  6,  9,  12, 15, 18, 21]
    temperatures = [14, 11, 9,  13, 24, 28, 23, 17]
    target       = 10.5

    print()
    print("+==================================================+")
    print("|        אינטרפולציה ספליין קובי - דוגמה          |")
    print("+==================================================+")
    print()
    print("נקודות המדידה הידועות:")
    print()
    print("  +-------------+------------------+")
    print("  |    שעה      |   טמפרטורה       |")
    print("  +-------------+------------------+")
    for hour, temp in zip(hours, temperatures):
        print(f"  |  {hour:02d}:00       |   {temp:>5.1f} מעלות    |")
    print("  +-------------+------------------+")

    print()
    print(f"שאלה: מה הייתה הטמפרטורה בשעה {target}?")
    print()

    result = cubic_spline(hours, temperatures, target)

    # מציאת הנקודות הסמוכות להקשר
    hour_before, temp_before, hour_after, temp_after = None, None, None, None
    for i in range(len(hours) - 1):
        if hours[i] <= target <= hours[i + 1]:
            hour_before, temp_before = hours[i], temperatures[i]
            hour_after,  temp_after  = hours[i + 1], temperatures[i + 1]
            break

    print("+==================================================+")
    print("|                    תוצאה                        |")
    print("+==================================================+")
    if hour_before is not None:
        print(f"|  שעה {hour_before:02d}:00  ->  {temp_before:.1f} מעלות  (מדידה ידועה)       |")
    print(f"|  שעה {target:.2f} ->  {result:.2f} מעלות  <- תוצאת האינטרפולציה  |")
    if hour_after is not None:
        print(f"|  שעה {hour_after:02d}:00  ->  {temp_after:.1f} מעלות  (מדידה ידועה)       |")
    print("+==================================================+")
    print()
    print(f"הטמפרטורה המשוערת בשעה {target} היא: {result:.2f} מעלות צלזיוס.")
    print()


if __name__ == "__main__":
    main()