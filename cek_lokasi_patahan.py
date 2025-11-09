import geopandas as gpd
from shapely.geometry import Point

# Load data patahan aktif (Hanya sekali saat script di-load)
try:
    gdf = gpd.read_file("patahan_aktif.geojson")    
    gdf = gdf.set_crs("EPSG:4326")
    gdf_m = gdf.to_crs("EPSG:3857") # Konversi ke meter sekali saja
    print("Data GeoJSON 'patahan_aktif.geojson' berhasil di-load.")
except Exception as e:
    print(f"!!! ERROR: Gagal load 'patahan_aktif.geojson'. Pastikan file ada. Error: {e}")
    gdf_m = None

def cek_lokasi_patahan(lat: float, lon: float, radius_km: int = 10):
    """
    Cek patahan terdekat dari titik lat/lon dan kembalikan hasilnya
    sebagai dictionary.
    """
    if gdf_m is None:
        return {
            "status": "ERROR: Data Patahan (GeoJSON) tidak berhasil di-load di server.",
            "nama_patahan": "Tidak diketahui",
            "jarak_km": -1
        }
        
    titik = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    titik_m = titik.to_crs("EPSG:3857") # Konversi titik ke meter

    # Hitung jarak minimum ke semua patahan
    gdf_m["jarak_m"] = gdf_m.geometry.distance(titik_m.geometry.iloc[0])
    
    # Ambil patahan terdekat
    patahan_terdekat = gdf_m.loc[gdf_m["jarak_m"].idxmin()]
    
    nama_patahan = patahan_terdekat['namobj']
    jarak_meter = patahan_terdekat['jarak_m']
    jarak_km = jarak_meter / 1000

    # Buat summary teks
    if jarak_km <= radius_km:
        status_text = (
            f"Status: ZONA PERINGATAN. "
            f"Lokasi Anda berada {jarak_km:.2f} km dari {nama_patahan}. "
            "Disarankan kewaspadaan tinggi dan inspeksi struktur."
        )
    else:
        status_text = (
            f"Status: AMAN (Relatif). "
            f"Patahan aktif terdekat adalah {nama_patahan}, "
            f"berjarak {jarak_km:.2f} km dari lokasi Anda (di luar radius {radius_km} km)."
        )

    # Kembalikan sebagai dictionary
    return {
        "status": status_text,
        "nama_patahan": nama_patahan,
        "jarak_km": round(jarak_km, 2)
    }

# ==========================================================
# BARU: Ini adalah pelindungnya
# Kode di bawah ini HANYA akan jalan jika kamu
# menjalankan `python3 cek_lokasi_patahan.py`
# dan TIDAK akan jalan saat di-import oleh app.py
# ==========================================================
if __name__ == "__main__":
    print("Menjalankan tes mandiri untuk 'cek_lokasi_patahan.py'...")
    # Contoh uji: misal lokasi hasil deteksi breksi
    hasil_tes = cek_lokasi_patahan(lat=-7.000928, lon=109.298526, radius_km=10)
    print(hasil_tes)
    
    print("\nMenjalankan tes lokasi aman...")
    # Contoh uji: Lokasi aman (misal: tengah laut)
    hasil_tes_aman = cek_lokasi_patahan(lat=-5.0, lon=110.0, radius_km=10)
    print(hasil_tes_aman)