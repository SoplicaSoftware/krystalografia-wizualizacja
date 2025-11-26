import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import TextBox, Button
import numpy as np
from itertools import combinations, product
from fractions import Fraction

# --- Zmienne globalne ---
current_h, current_k, current_l = 0, -1, 3
step = 0 

# --- Konfiguracja okna ---
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(bottom=0.1, left=0.02, right=0.98, wspace=0.1)

# Lewa strona: Wykres 3D
ax = fig.add_subplot(1, 2, 1, projection='3d')

# Prawa strona: Panel Instrukcji
ax_instr = fig.add_subplot(1, 2, 2)
ax_instr.axis('off')

def to_frac_latex(val):
    """
    Zamienia liczbę na ułamek LaTeX (BEZ znaków $).
    """
    if val == 0: return "0"
    f = Fraction(val).limit_denominator(20)
    if f.denominator == 1:
        return str(f.numerator)
    return r"\frac{" + str(f.numerator) + r"}{" + str(f.denominator) + r"}"

def index_to_math(n):
    """
    Zwraca łańcuch z reprezentacją indeksu w mathtext (np. liczba ujemna
    jako nadpisana kreską, używamy \overline{} aby zadziałało dla wielocyfrowych).
    Zwraca string zawierający delimitery dolara, np. "$\overline{1}$" lub "$2$".
    """
    if n == 0:
        return "$0$"
    if n < 0:
        return r"$\overline{" + str(abs(n)) + r"}$"
    return f"${n}$"

def draw_base_cube():
    """Rysuje bazę sześcianu."""
    ax.cla()
    r = [0, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="#dddddd", linewidth=1)
            
    # Stałe zero
    ax.scatter(0, 0, 0, color='black', s=40)
    ax.text(0, 0, -0.08, "(0,0,0)", fontsize=8)
    
    # Widok
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-35)
    
    # Osie główne
    len_ax = 1.1
    ax.quiver(0,0,0, len_ax,0,0, color='red', arrow_length_ratio=0.05, alpha=0.3)
    ax.text(len_ax, 0, 0, "x", color='red')
    ax.quiver(0,0,0, 0,len_ax,0, color='green', arrow_length_ratio=0.05, alpha=0.3)
    ax.text(0, len_ax, 0, "y", color='green')
    ax.quiver(0,0,0, 0,0,len_ax, color='blue', arrow_length_ratio=0.05, alpha=0.3)
    ax.text(0, 0, len_ax, "z", color='blue')

def calculate_logic(h, k, l):
    # Punkt startowy
    ox = 1 if h < 0 else 0
    oy = 1 if k < 0 else 0
    oz = 1 if l < 0 else 0
    start = [ox, oy, oz]
    
    # Punkty na osiach
    points = []
    
    # X
    px = None
    if h != 0:
        dist = 1.0/abs(h)
        direction = -1 if h < 0 else 1
        px = [ox + direction*dist, oy, oz]
        if -0.01 <= px[0] <= 1.01: points.append(px)
        
    # Y
    py = None
    if k != 0:
        dist = 1.0/abs(k)
        direction = -1 if k < 0 else 1
        py = [ox, oy + direction*dist, oz]
        if -0.01 <= py[1] <= 1.01: points.append(py)
        
    # Z
    pz = None
    if l != 0:
        dist = 1.0/abs(l)
        direction = -1 if l < 0 else 1
        pz = [ox, oy, oz + direction*dist]
        if -0.01 <= pz[2] <= 1.01: points.append(pz)
        
    return start, points, (px, py, pz)

def draw_instruction_panel(h, k, l):
    """Rysuje tekst i tabele."""
    ax_instr.clear()
    ax_instr.axis('off')
    
    # Nagłówek
    # Używamy mathtext, aby dla wartości ujemnych pokazać kreskę nad liczbą
    header = "Wskaźniki: (" + index_to_math(h) + " " + index_to_math(k) + " " + index_to_math(l) + ")"
    ax_instr.text(0.5, 0.95, header, ha='center', fontsize=16, fontweight='bold', transform=ax_instr.transAxes)

    if step == 0:
        ax_instr.text(0.5, 0.6, "Kliknij 'DALEJ >', aby rozpocząć\nrysowanie krok po kroku.", 
                      ha='center', fontsize=12, transform=ax_instr.transAxes)

    elif step == 1:
        # KROK 1
        ax_instr.text(0.05, 0.85, "KROK 1: Punkt Startowy (Baza)", fontsize=14, fontweight='bold', color='orange', transform=ax_instr.transAxes)
        
        lines = ["Analiza znaków:"]
        shifts = []
        if h < 0: lines.append(f"• X = {h} (ujemne) -> Start X=1"); shifts.append("X=1")
        if k < 0: lines.append(f"• Y = {k} (ujemne) -> Start Y=1"); shifts.append("Y=1")
        if l < 0: lines.append(f"• Z = {l} (ujemne) -> Start Z=1"); shifts.append("Z=1")
        
        if not shifts:
            lines.append("• Wszystkie dodatnie -> Start (0,0,0)")
            res = "Startujesz w (0,0,0) [Czarny punkt]"
        else:
            res = f"Startujesz w narożniku: {', '.join(shifts)}\n[Pomarańczowy punkt]"
            
        ax_instr.text(0.05, 0.75, "\n".join(lines), fontsize=12, transform=ax_instr.transAxes, va='top')
        ax_instr.text(0.05, 0.50, res, fontsize=12, fontweight='bold', bbox=dict(facecolor='orange', alpha=0.2), transform=ax_instr.transAxes)

    elif step == 2:
        # KROK 2: Tabela
        ax_instr.text(0.05, 0.85, "KROK 2: Obliczanie punktów", fontsize=14, fontweight='bold', color='blue', transform=ax_instr.transAxes)
        
        col_labels = ['Wskaźnik', 'Równanie (Odwrotność)']
        table_vals = []
        
        # Oś X
        if h == 0:
            math_x = r"$1/x = 0 \rightarrow \infty$"
        else:
            res = to_frac_latex(1/abs(h))
            math_x = fr"$1/x = {abs(h)} \rightarrow {res}$"
        table_vals.append([f"h={h}", math_x])

        # Oś Y
        if k == 0:
            math_y = r"$1/y = 0 \rightarrow \infty$"
        else:
            res = to_frac_latex(1/abs(k))
            math_y = fr"$1/y = {abs(k)} \rightarrow {res}$"
        table_vals.append([f"k={k}", math_y])
        
        # Oś Z
        if l == 0:
            math_z = r"$1/z = 0 \rightarrow \infty$"
        else:
            res = to_frac_latex(1/abs(l))
            math_z = fr"$1/z = {abs(l)} \rightarrow {res}$"
        table_vals.append([f"l={l}", math_z])
        
        # Rysowanie tabeli
        the_table = ax_instr.table(cellText=table_vals, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.05, 0.5, 0.9, 0.3])
        the_table.auto_set_font_size(False); the_table.set_fontsize(14); the_table.scale(1, 2.5)
        
        ax_instr.text(0.05, 0.4, "Wynik oznacza odległość od POMARAŃCZOWEGO punktu.", color='red', fontsize=11, transform=ax_instr.transAxes)

    elif step == 3:
        # KROK 3
        ax_instr.text(0.05, 0.85, "KROK 3: Łączenie / Równoległość", fontsize=14, fontweight='bold', color='purple', transform=ax_instr.transAxes)
        zeros = [h, k, l].count(0)
        lines = []
        if zeros == 0:
            lines.append("Brak zer -> TRÓJKĄT.")
            lines.append("Połącz 3 punkty wyznaczone w kroku 2.")
        elif zeros == 1:
            zero_axis = "X" if h==0 else ("Y" if k==0 else "Z")
            lines.append(f"Wskaźnik {zero_axis} = 0 (Nieskończoność).")
            lines.append(f"Płaszczyzna RÓWNOLEGŁA do osi {zero_axis}.")
            lines.append(f"1. Połącz dwa punkty.")
            lines.append(f"2. Rozciągnij wzdłuż osi {zero_axis}.")
        elif zeros == 2:
             non_zero = "X" if h!=0 else ("Y" if k!=0 else "Z")
             lines.append(f"Tylko wskaźnik {non_zero} jest liczbą.")
             lines.append(f"Płaszczyzna przecina oś {non_zero} w wyliczonym punkcie.")
             lines.append("Jest to ściana prostopadła do tej osi.")
        else:
            lines.append("To cały sześcian?")
        ax_instr.text(0.05, 0.70, "\n".join(lines), fontsize=12, transform=ax_instr.transAxes, va='top', linespacing=1.8)

    elif step == 4:
        # KROK 4
        ax_instr.text(0.05, 0.85, "KROK 4: Gotowe!", fontsize=14, fontweight='bold', color='green', transform=ax_instr.transAxes)
        ax_instr.text(0.05, 0.70, "Oto Twoja płaszczyzna.\nMożesz wpisać nowe liczby na dole i kliknąć 'Załaduj'.", fontsize=12, transform=ax_instr.transAxes)

def draw_scene():
    draw_base_cube()
    h, k, l = current_h, current_k, current_l
    start, pts, raw_pts = calculate_logic(h, k, l)
    ox, oy, oz = start
    
    # KROK 1: Start
    if step >= 1:
        color = 'orange'
        ax.scatter(ox, oy, oz, color=color, s=100, edgecolors='red', zorder=10)
        # Lokalne osie
        d = 0.4
        sx = -1 if h<0 else 1; sy = -1 if k<0 else 1; sz = -1 if l<0 else 1
        ax.quiver(ox,oy,oz, sx*d,0,0, color='orange', linestyle='--', alpha=0.5)
        ax.quiver(ox,oy,oz, 0,sy*d,0, color='orange', linestyle='--', alpha=0.5)
        ax.quiver(ox,oy,oz, 0,0,sz*d, color='orange', linestyle='--', alpha=0.5)

    # KROK 2: Punkty
    if step >= 2:
        px, py, pz = raw_pts
        if px: ax.scatter(*px, color='red', s=50); ax.plot([ox, px[0]], [oy, px[1]], [oz, px[2]], color='red')
        if py: ax.scatter(*py, color='green', s=50); ax.plot([ox, py[0]], [oy, py[1]], [oz, py[2]], color='green')
        if pz: ax.scatter(*pz, color='blue', s=50); ax.plot([ox, pz[0]], [oy, pz[1]], [oz, pz[2]], color='blue')

    # KROK 3: Linie Równoległości
    final_poly_pts = []
    zeros = [h, k, l].count(0)
    base_pts = [p for p in pts]
    
    if step >= 3:
        if zeros == 1:
            idx = 0 if h==0 else (1 if k==0 else 2)
            for p in base_pts:
                p1 = list(p); p1[idx] = 0
                p2 = list(p); p2[idx] = 1
                final_poly_pts.append(p1); final_poly_pts.append(p2)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='purple', linestyle='--', linewidth=1)
        elif zeros == 0:
            final_poly_pts = base_pts
        elif zeros == 2:
            # Płaszczyzna typu ściana (np. 2 0 0)
            if h != 0: 
                # Cięcie na osi X w punkcie px[0]
                x_val = raw_pts[0][0]
                final_poly_pts = [[x_val, 0, 0], [x_val, 1, 0], [x_val, 1, 1], [x_val, 0, 1]]
            elif k != 0:
                y_val = raw_pts[1][1]
                final_poly_pts = [[0, y_val, 0], [1, y_val, 0], [1, y_val, 1], [0, y_val, 1]]
            elif l != 0:
                z_val = raw_pts[2][2]
                final_poly_pts = [[0, 0, z_val], [1, 0, z_val], [1, 1, z_val], [0, 1, z_val]]
            
            # Rysuj ramkę ściany
            if step == 3: # Tylko ramka
                p = final_poly_pts
                # Zamknięta pętla
                loop = p + [p[0]]
                xs, ys, zs = zip(*loop)
                ax.plot(xs, ys, zs, color='purple', linestyle='--', linewidth=1.5)

    # KROK 4: Wypełnienie
    if step == 4 and len(final_poly_pts) >= 3:
        center = np.mean(np.array(final_poly_pts), axis=0)
        norm = np.array([h, k, l]) if h!=0 or k!=0 or l!=0 else np.array([1,0,0])
        try:
            z_ax = norm / np.linalg.norm(norm)
            tmp = np.array([1,0,0]) if abs(z_ax[0]) < 0.9 else np.array([0,1,0])
            x_ax = np.cross(z_ax, tmp); x_ax /= np.linalg.norm(x_ax)
            y_ax = np.cross(z_ax, x_ax)
            sorted_pts = sorted(final_poly_pts, key=lambda p: np.arctan2(np.dot(p-center, y_ax), np.dot(p-center, x_ax)))
            poly = Poly3DCollection([sorted_pts], alpha=0.6, facecolors='cyan', edgecolors='blue')
            ax.add_collection3d(poly)
        except: pass

    draw_instruction_panel(h, k, l)
    fig.canvas.draw_idle()

# --- Sterowanie ---
def update_step(val):
    global step
    step += val
    if step < 0: step = 0
    if step > 4: step = 4
    draw_scene()

def submit(val):
    global current_h, current_k, current_l, step
    try:
        current_h = int(tb_h.text)
        current_k = int(tb_k.text)
        current_l = int(tb_l.text)
        step = 0
        draw_scene()
    except ValueError:
        print("Błąd: Wpisz liczby całkowite!")

# UI
axbox_h = plt.axes([0.1, 0.02, 0.05, 0.05]); tb_h = TextBox(axbox_h, 'h:', initial="0")
axbox_k = plt.axes([0.2, 0.02, 0.05, 0.05]); tb_k = TextBox(axbox_k, 'k:', initial="-1")
axbox_l = plt.axes([0.3, 0.02, 0.05, 0.05]); tb_l = TextBox(axbox_l, 'l:', initial="3")

tb_h.on_submit(submit)
tb_k.on_submit(submit)
tb_l.on_submit(submit)

btn_start = Button(plt.axes([0.4, 0.02, 0.1, 0.05]), 'Załaduj')
btn_start.on_clicked(submit)

btn_prev = Button(plt.axes([0.6, 0.02, 0.1, 0.05]), '< Wstecz')
btn_prev.on_clicked(lambda x: update_step(-1))

btn_next = Button(plt.axes([0.75, 0.02, 0.15, 0.05]), 'DALEJ >', color='lightgreen')
btn_next.on_clicked(lambda x: update_step(1))

# Start
step = 0
draw_scene()
plt.show()