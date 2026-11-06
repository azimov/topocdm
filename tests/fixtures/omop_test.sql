-- Minimal OMOP CDM for the integration test. Five patients; patient 1 has two
-- observation periods, which is the case that broke the Databricks fit (#1).
drop schema if exists cdm cascade;
create schema cdm;

create table cdm.person (person_id bigint primary key);
create table cdm.observation_period (
    person_id bigint not null,
    observation_period_start_date date not null,
    observation_period_end_date date not null
);
create table cdm.condition_occurrence (
    person_id bigint, condition_concept_id integer,
    condition_start_date date, condition_end_date date
);
create table cdm.drug_exposure (
    person_id bigint, drug_concept_id integer,
    drug_exposure_start_date date, drug_exposure_end_date date
);
create table cdm.visit_occurrence (
    person_id bigint, visit_concept_id integer,
    visit_start_date date, visit_end_date date
);
create table cdm.procedure_occurrence (
    person_id bigint, procedure_concept_id integer, procedure_date date
);
create table cdm.measurement (
    person_id bigint, measurement_concept_id integer, measurement_date date
);
create table cdm.observation (
    person_id bigint, observation_concept_id integer, observation_date date
);
create table cdm.death (person_id bigint, death_date date);

insert into cdm.person select generate_series(1, 5);

insert into cdm.observation_period values
  (1, date '2020-01-01', date '2020-12-31'),
  (1, date '2021-01-01', date '2022-12-31'),
  (2, date '2020-01-01', date '2022-12-31'),
  (3, date '2020-01-01', date '2022-12-31'),
  (4, date '2020-01-01', date '2022-12-31'),
  (5, date '2020-01-01', date '2022-12-31');

insert into cdm.condition_occurrence
select p.person_id, 100 + (g % 3),
       date '2020-01-01' + ((g * 17) % 900),
       date '2020-01-01' + ((g * 17) % 900) + 30
  from generate_series(1, 60) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.drug_exposure
select p.person_id, 200 + (g % 3),
       date '2020-01-01' + ((g * 23) % 900),
       date '2020-01-01' + ((g * 23) % 900) + 60
  from generate_series(1, 60) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.visit_occurrence
select p.person_id, 9201,
       date '2020-01-01' + ((g * 31) % 900),
       date '2020-01-01' + ((g * 31) % 900)
  from generate_series(1, 10) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.procedure_occurrence
select p.person_id, 300 + (g % 2),
       date '2020-01-01' + ((g * 41) % 900)
  from generate_series(1, 30) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.measurement
select p.person_id, 400 + (g % 2),
       date '2020-01-01' + ((g * 43) % 900)
  from generate_series(1, 40) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.observation
select p.person_id, 500 + (g % 2),
       date '2020-01-01' + ((g * 47) % 900)
  from generate_series(1, 20) g
  cross join generate_series(1, 5) as p(person_id);

insert into cdm.death values (5, date '2022-06-01');
