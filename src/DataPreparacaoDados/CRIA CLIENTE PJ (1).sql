CREATE DATABASE [CLIENTE_HOLDING];



USE [CLIENTE_HOLDING];




create table CLIENTE_PJ
(
  CNPJ VARCHAR(40) not null PRIMARY KEY,
  NOMECLIENTE      VARCHAR(40),
  EMAILCLIENTE     VARCHAR(50),
  TELEFONECLIENTE  VARCHAR(12),
  DESCRICAOCLIENTE VARCHAR(300),
  CEPCLIENTE       VARCHAR(9),
  RUA              VARCHAR(30),
  NUMEROEND        INTEGER,
  COMPLEMENTOEND   VARCHAR(20),
  CIDADECLIENTE    VARCHAR(40),
  ESTADO           VARCHAR(40)
)

DELETE  CLIENTE_PJ

insert into cliente_pj 
values ('  5511111000101  ', 'Carandai LTDA', 'LOJA2@GMAIL.COM', '01125254691', 'ATACADO-PJ', '02281-040', 'Rua Matos', 26, '', 'GUARULHOS', 'sp');

insert into cliente_pj 
values ('  5522222000202', 'Vitória Company', 'LOJA3@GMAIL.COM', '01125896347', 'ATACADISTA-PJ', '13590-000', 'Rod. Luiz Augusto', 40, '', 'DOURADO', 'sp');

insert into cliente_pj 
values ('     5533333000303', 'Loja AXE', 'AXE@YAHOO.COM.BR', '08821459853', 'ATACADO-PJ', '62010-020', 'Rua Luzanir Coelho', 23, '', 'Sobral', 'ce');

insert into cliente_pj 
values ('5544444000404', 'Assis LTDA', 'ASS@ASSIS.COM.B', '06325879412', 'ATACADISTA-PJ', '77001-016', '        Quadra 103 Norte', 55, '', 'Braspilia', 'df');

insert into cliente_pj 
values ('5566666000606', 'Tem Tudo', 'LOJA1@GMAIL.COM', '03132104000', 'ATACADO-PJ', '35900-070', 'RUA  PENHA', 185, '', 'ITABIRA', 'mg');

insert into cliente_pj 
values ('5555555000505', 'Alvares e Filhos', 'aff@GMAIL.COM', '03158963425', 'ATACADISTA-PJ', '30112-011', 'Avenida Brasil', 1781, '', 'BELO HORIZONTE', 'mg');

insert into cliente_pj 
values ('       5577777000707  ', '', '', '', 'ATACADISTA-PJ', '70070-130', 'SBS Quadra 3', 23, '', 'Brasília', 'df');

insert into cliente_pj 
values ('5588888000808  ', 'Ave Luz', 'aluz@aveluz.br', '08225479631', 'ATACADO-PJ', '57660-010', 'rua alto cruzeiro', 41, '', 'Anadia', 'al');
