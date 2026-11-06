-- A five-patient OMOP slice for the in-process DuckDB integration test.
--
-- Patient 1 has two observation periods, which is the multiple-period case that
-- broke the first warehouse fit. The event tables are generated with DuckDB's
-- range() so the fixture stays short and the intended shape is visible: for
-- each patient, 20 conditions, 20 drugs, 5 visits, 10 procedures, 15
-- measurements and 10 observations, spread over the observation window.

create table person (person_id bigint);
insert into person
values (1), (2), (3), (4), (5);

create table observation_period (
    person_id bigint,
    observation_period_start_date date,
    observation_period_end_date date
);
insert into observation_period
values
    (1, date '2020-01-01', date '2020-12-31'),
    (1, date '2021-01-01', date '2022-12-31'),
    (2, date '2020-01-01', date '2022-12-31'),
    (3, date '2020-01-01', date '2022-12-31'),
    (4, date '2020-01-01', date '2022-12-31'),
    (5, date '2020-01-01', date '2022-12-31');

create table condition_occurrence as
select p.person_id,
       (100 + g % 3)::integer as condition_concept_id,
       (date '2020-01-01' + (((g * 17) % 900)::integer))::date as condition_start_date,
       (date '2020-01-01' + (((g * 17) % 900)::integer) + 30)::date as condition_end_date
from range(1, 6) as p(person_id), range(0, 20) as g(g);

create table drug_exposure as
select p.person_id,
       (200 + g % 3)::integer as drug_concept_id,
       (date '2020-01-01' + (((g * 23) % 900)::integer))::date as drug_exposure_start_date,
       (date '2020-01-01' + (((g * 23) % 900)::integer) + 60)::date as drug_exposure_end_date
from range(1, 6) as p(person_id), range(0, 20) as g(g);

create table visit_occurrence as
select p.person_id,
       9201::integer as visit_concept_id,
       (date '2020-01-01' + (((g * 31) % 900)::integer))::date as visit_start_date,
       (date '2020-01-01' + (((g * 31) % 900)::integer))::date as visit_end_date
from range(1, 6) as p(person_id), range(0, 5) as g(g);

create table procedure_occurrence as
select p.person_id,
       (300 + g % 2)::integer as procedure_concept_id,
       (date '2020-01-01' + (((g * 41) % 900)::integer))::date as procedure_date
from range(1, 6) as p(person_id), range(0, 10) as g(g);

create table measurement as
select p.person_id,
       (400 + g % 2)::integer as measurement_concept_id,
       (date '2020-01-01' + (((g * 43) % 900)::integer))::date as measurement_date
from range(1, 6) as p(person_id), range(0, 15) as g(g);

create table observation as
select p.person_id,
       (500 + g % 2)::integer as observation_concept_id,
       (date '2020-01-01' + (((g * 47) % 900)::integer))::date as observation_date
from range(1, 6) as p(person_id), range(0, 10) as g(g);

create table death (person_id bigint, death_date date);
insert into death values (5, date '2022-06-01');
