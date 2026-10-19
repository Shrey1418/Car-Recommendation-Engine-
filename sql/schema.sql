DROP TABLE IF EXISTS variants;

CREATE TABLE variants (
    variant_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    brand                TEXT    NOT NULL,
    model                TEXT    NOT NULL,
    variant              TEXT    NOT NULL,
    fuel                 TEXT    NOT NULL,
    transmission         TEXT    NOT NULL CHECK (transmission IN ('Manual', 'Automatic')),
    transmission_subtype TEXT,
    ex_showroom_price    REAL    NOT NULL CHECK (ex_showroom_price > 0),
    engine_cc            REAL,
    claimed_mileage      REAL,
    mileage_source       TEXT,
    body_type            TEXT    NOT NULL,
    seats                INTEGER NOT NULL,
    power_ps             REAL,
    torque_nm            REAL,
    cylinders            REAL,
    fuel_tank_l          REAL,
    length_mm            REAL,
    width_mm             REAL,
    height_mm            REAL,
    spec_source          TEXT,
    ev_range_km          REAL,
    battery_kwh          REAL,
    ev_range_basis       TEXT,
    UNIQUE (brand, model, variant, fuel, transmission)
);

CREATE INDEX idx_variants_price ON variants (ex_showroom_price);
CREATE INDEX idx_variants_body  ON variants (body_type, seats);