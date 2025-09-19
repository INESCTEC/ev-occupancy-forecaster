LICENÇA PÚBLICA DA UNIÃO EUROPEIA v. 1.2 
EUPL © União Europeia 2007, 2016 
A presente licença pública da União Europeia («EUPL») aplica-se a qualquer obra (como definido abaixo) que seja fornecido nos termos da presente licença. É proibida qualquer utilização da obra diferente da autorizada nos termos da presente licença (na medida em que tal utilização esteja abrangida por um direito do titular dos direitos de autor da obra). A obra é fornecida nos termos da presente licença caso o licenciante (como definido abaixo) tenha colocado o seguinte aviso imediatamente após a informação sobre os direitos de autor respeitantes à obra: 
                          Licenciada nos termos da EUPL 
ou tenha expresso por qualquer outro meio a sua vontade de licenciar a obra nos termos da EUPL.
 
1.Definições Copyright (C) <2025>, INESC TEC 

VoltTune: Greedy Voltage Correction Using Flexibility Allocation

This software is authored by:
José Pedro Ribeiro de Sousa
Gil Silva Sampaio

This program is free software: you can redistribute it and/or modify it under the terms of the European Union Public License v1.2 as published by the European Union.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the European Union Public License v1.2 for more details.
You should have received a copy of the European Union Public License v1.2 along with this program. If not, see https://www.eupl.eu

This program makes use and is distributed with the following libraries:
* numpy: https://github.com/numpy/numpy
* os: https://github.com/numpy/numpy

A commercial license is also available for use in industrial projects and collaborations that do not wish to use the EUPL v1.2 license. 
To obtain the commercial license please contact the INESC TEC Technology Licensing Office (TLO) at info.sal@inesctec.pt, or 
Campus da Faculdade de Engenharia da Universidade do Porto 
Rua Dr. Roberto Frias 
4200-465 Porto 
Portugal 

If needed SAL (INESC TEC Technology Licensing Office - TLO) can assist with all the legal details regarding the licensing agreement 
 
If you use VoltTune in a work that leads to a scientific publication, we would appreciate it if you would kindly cite VoltTune in your manuscript. 
 
 EUROPEAN UNION PUBLIC LICENCE v. 1.2
                      EUPL © the European Union 2007, 2016

This European Union Public Licence (the ‘EUPL’) applies to the Work (as defined
below) which is provided under the terms of this Licence. Any use of the Work,
other than as authorised under this Licence is prohibited (to the extent such
use is covered by a right of the copyright holder of the Work).

The Work is provided under the terms of this Licence when the Licensor (as
defined below) has placed the following notice immediately following the
copyright notice for the Work:

        Licensed under the EUPL

or has expressed by any other means his willingness to license under the EUPL.

1. Definitions

In this Licence, the following terms have the following meaning:

- ‘The Licence’: this Licence.

- ‘The Original Work’: the work or software distributed or communicated by the
  Licensor under this Licence, available as Source Code and also as Executable
  Code as the case may be.

- ‘Derivative Works’: the works or software that could be created by the
  Licensee, based upon the Original Work or modifications thereof. This Licence
  does not define the extent of modification or dependence on the Original Work
  required in order to classify a work as a Derivative Work; this extent is
  determined by copyright law applicable in the country mentioned in Article 15.

- ‘The Work’: the Original Work or its Derivative Works.

- ‘The Source Code’: the human-readable form of the Work which is the most
  convenient for people to study and modify.

- ‘The Executable Code’: any code which has generally been compiled and which is
  meant to be interpreted by a computer as a program.

- ‘The Licensor’: the natural or legal person that distributes or communicates
  the Work under the Licence.

- ‘Contributor(s)’: any natural or legal person who modifies the Work under the
  Licence, or otherwise contributes to the creation of a Derivative Work.

- ‘The Licensee’ or ‘You’: any natural or legal person who makes any usage of
  the Work under the terms of the Licence.

- ‘Distribution’ or ‘Communication’: any act of selling, giving, lending,
  renting, distributing, communicating, transmitting, or otherwise making
  available, online or offline, copies of the Work or providing access to its
  essential functionalities at the disposal of any other natural or legal
  person.

2. Scope of the rights granted by the Licence

The Licensor hereby grants You a worldwide, royalty-free, non-exclusive,
sublicensable licence to do the following, for the duration of copyright vested
in the Original Work:

- use the Work in any circumstance and for all usage,
- reproduce the Work,
- modify the Work, and make Derivative Works based upon the Work,
- communicate to the public, including the right to make available or display
  the Work or copies thereof to the public and perform publicly, as the case may
  be, the Work,
- distribute the Work or copies thereof,
- lend and rent the Work or copies thereof,
- sublicense rights in the Work or copies thereof.

Those rights can be exercised on any media, supports and formats, whether now
known or later invented, as far as the applicable law permits so.

In the countries where moral rights apply, the Licensor waives his right to
exercise his moral right to the extent allowed by law in order to make effective
the licence of the economic rights here above listed.

The Licensor grants to the Licensee royalty-free, non-exclusive usage rights to
any patents held by the Licensor, to the extent necessary to make use of the
rights granted on the Work under this Licence.

3. Communication of the Source Code

The Licensor may provide the Work either in its Source Code form, or as
Executable Code. If the Work is provided as Executable Code, the Licensor
provides in addition a machine-readable copy of the Source Code of the Work
along with each copy of the Work that the Licensor distributes or indicates, in
a notice following the copyright notice attached to the Work, a repository where
the Source Code is easily and freely accessible for as long as the Licensor
continues to distribute or communicate the Work.

4. Limitations on copyright

Nothing in this Licence is intended to deprive the Licensee of the benefits from
any exception or limitation to the exclusive rights of the rights owners in the
Work, of the exhaustion of those rights or of other applicable limitations
thereto.

5. Obligations of the Licensee

The grant of the rights mentioned above is subject to some restrictions and
obligations imposed on the Licensee. Those obligations are the following:

Attribution right: The Licensee shall keep intact all copyright, patent or
trademarks notices and all notices that refer to the Licence and to the
disclaimer of warranties. The Licensee must include a copy of such notices and a
copy of the Licence with every copy of the Work he/she distributes or
communicates. The Licensee must cause any Derivative Work to carry prominent
notices stating that the Work has been modified and the date of modification.

Copyleft clause: If the Licensee distributes or communicates copies of the
Original Works or Derivative Works, this Distribution or Communication will be
done under the terms of this Licence or of a later version of this Licence
unless the Original Work is expressly distributed only under this version of the
Licence — for example by communicating ‘EUPL v. 1.2 only’. The Licensee
(becoming Licensor) cannot offer or impose any additional terms or conditions on
the Work or Derivative Work that alter or restrict the terms of the Licence.

Compatibility clause: If the Licensee Distributes or Communicates Derivative
Works or copies thereof based upon both the Work and another work licensed under
a Compatible Licence, this Distribution or Communication can be done under the
terms of this Compatible Licence. For the sake of this clause, ‘Compatible
Licence’ refers to the licences listed in the appendix attached to this Licence.
Should the Licensee's obligations under the Compatible Licence conflict with
his/her obligations under this Licence, the obligations of the Compatible
Licence shall prevail.

Provision of Source Code: When distributing or communicating copies of the Work,
the Licensee will provide a machine-readable copy of the Source Code or indicate
a repository where this Source will be easily and freely available for as long
as the Licensee continues to distribute or communicate the Work.

Legal Protection: This Licence does not grant permission to use the trade names,
trademarks, service marks, or names of the Licensor, except as required for
reasonable and customary use in describing the origin of the Work and
reproducing the content of the copyright notice.

6. Chain of Authorship

The original Licensor warrants that the copyright in the Original Work granted
hereunder is owned by him/her or licensed to him/her and that he/she has the
power and authority to grant the Licence.

Each Contributor warrants that the copyright in the modifications he/she brings
to the Work are owned by him/her or licensed to him/her and that he/she has the
power and authority to grant the Licence.

Each time You accept the Licence, the original Licensor and subsequent
Contributors grant You a licence to their contributions to the Work, under the
terms of this Licence.

7. Disclaimer of Warranty

The Work is a work in progress, which is continuously improved by numerous
Contributors. It is not a finished work and may therefore contain defects or
‘bugs’ inherent to this type of development.

For the above reason, the Work is provided under the Licence on an ‘as is’ basis
and without warranties of any kind concerning the Work, including without
limitation merchantability, fitness for a particular purpose, absence of defects
or errors, accuracy, non-infringement of intellectual property rights other than
copyright as stated in Article 6 of this Licence.

This disclaimer of warranty is an essential part of the Licence and a condition
for the grant of any rights to the Work.

8. Disclaimer of Liability

Except in the cases of wilful misconduct or damages directly caused to natural
persons, the Licensor will in no event be liable for any direct or indirect,
material or moral, damages of any kind, arising out of the Licence or of the use
of the Work, including without limitation, damages for loss of goodwill, work
stoppage, computer failure or malfunction, loss of data or any commercial
damage, even if the Licensor has been advised of the possibility of such damage.
However, the Licensor will be liable under statutory product liability laws as
far such laws apply to the Work.

9. Additional agreements

While distributing the Work, You may choose to conclude an additional agreement,
defining obligations or services consistent with this Licence. However, if
accepting obligations, You may act only on your own behalf and on your sole
responsibility, not on behalf of the original Licensor or any other Contributor,
and only if You agree to indemnify, defend, and hold each Contributor harmless
for any liability incurred by, or claims asserted against such Contributor by
the fact You have accepted any warranty or additional liability.

10. Acceptance of the Licence

The provisions of this Licence can be accepted by clicking on an icon ‘I agree’
placed under the bottom of a window displaying the text of this Licence or by
affirming consent in any other similar way, in accordance with the rules of
applicable law. Clicking on that icon indicates your clear and irrevocable
acceptance of this Licence and all of its terms and conditions.

Similarly, you irrevocably accept this Licence and all of its terms and
conditions by exercising any rights granted to You by Article 2 of this Licence,
such as the use of the Work, the creation by You of a Derivative Work or the
Distribution or Communication by You of the Work or copies thereof.

11. Information to the public

In case of any Distribution or Communication of the Work by means of electronic
communication by You (for example, by offering to download the Work from a
remote location) the distribution channel or media (for example, a website) must
at least provide to the public the information requested by the applicable law
regarding the Licensor, the Licence and the way it may be accessible, concluded,
stored and reproduced by the Licensee.

12. Termination of the Licence

The Licence and the rights granted hereunder will terminate automatically upon
any breach by the Licensee of the terms of the Licence.

Such a termination will not terminate the licences of any person who has
received the Work from the Licensee under the Licence, provided such persons
remain in full compliance with the Licence.

13. Miscellaneous

Without prejudice of Article 9 above, the Licence represents the complete
agreement between the Parties as to the Work.

If any provision of the Licence is invalid or unenforceable under applicable
law, this will not affect the validity or enforceability of the Licence as a
whole. Such provision will be construed or reformed so as necessary to make it
valid and enforceable.

The European Commission may publish other linguistic versions or new versions of
this Licence or updated versions of the Appendix, so far this is required and
reasonable, without reducing the scope of the rights granted by the Licence. New
versions of the Licence will be published with a unique version number.

All linguistic versions of this Licence, approved by the European Commission,
have identical value. Parties can take advantage of the linguistic version of
their choice.

14. Jurisdiction

Without prejudice to specific agreement between parties,

- any litigation resulting from the interpretation of this License, arising
  between the European Union institutions, bodies, offices or agencies, as a
  Licensor, and any Licensee, will be subject to the jurisdiction of the Court
  of Justice of the European Union, as laid down in article 272 of the Treaty on
  the Functioning of the European Union,

- any litigation arising between other parties and resulting from the
  interpretation of this License, will be subject to the exclusive jurisdiction
  of the competent court where the Licensor resides or conducts its primary
  business.

15. Applicable Law

Without prejudice to specific agreement between parties,

- this Licence shall be governed by the law of the European Union Member State
  where the Licensor has his seat, resides or has his registered office,

- this licence shall be governed by Belgian law if the Licensor has no seat,
  residence or registered office inside a European Union Member State.

Appendix

‘Compatible Licences’ according to Article 5 EUPL are:

- GNU General Public License (GPL) v. 2, v. 3
- GNU Affero General Public License (AGPL) v. 3
- Open Software License (OSL) v. 2.1, v. 3.0
- Eclipse Public License (EPL) v. 1.0
- CeCILL v. 2.0, v. 2.1
- Mozilla Public Licence (MPL) v. 2
- GNU Lesser General Public Licence (LGPL) v. 2.1, v. 3
- Creative Commons Attribution-ShareAlike v. 3.0 Unported (CC BY-SA 3.0) for
  works other than software
- European Union Public Licence (EUPL) v. 1.1, v. 1.2
- Québec Free and Open-Source Licence — Reciprocity (LiLiQ-R) or Strong
  Reciprocity (LiLiQ-R+).

The European Commission may update this Appendix to later versions of the above
licences without producing a new version of the EUPL, as long as they provide
the rights granted in Article 2 of this Licence and protect the covered Source
Code from exclusive appropriation.

All other changes or additions to this Appendix require the production of a new
EUPL version.

Na presente licença, entende-se por:  
— Licença:a presente licença.  
— Obra original:a obra ou software distribuídos ou comunicados pelo licenciante nos termos da presente licença, disponível como código-fonte e também como código executável, consoante o caso.  
— Obras derivadas:obras ou software que podem ser criados pelo licenciado com base na obra original ou mediante modificações nela introduzidas. A presente licença não define o grau de modificação ou de dependência em relação à obra original necessário para classificar uma obra como obra derivada; esse grau é determinado pela legislação relativa aos direitos de autor aplicável no país mencionado no ponto 15.  
— Obra:a obra original ou as obras dela derivadas.  
— Código-fonte:o formato legível por pessoas da obra que é o mais adequado para ser examinado e modificado.  
— Código executável:qualquer código que, em geral, foi compilado e se destina a ser interpretado por um computador como um programa.  
— Licenciante:pessoa singular ou coletiva que distribui ou comunica a obra nos termos da licença.  
— Contribuidor(es):pessoa singular ou coletiva que modifica a obra abrangida pela licença ou contribui de outro modo para a criação de uma obra derivada.  
— Licenciado:pessoa singular ou coletiva que utiliza a obra nos termos da licença.  — Distribuição ou comunicação:ato de venda, doação, comodato, locação, distribuição, comunicação, transmissão ou qualquer outra forma de disponibilização, em linha ou fora de linha, de cópias da obra ou do acesso às suas funcionalidades essenciais a qualquer outra pessoa singular ou coletiva. 

2.Âmbito dos direitos concedidos pela licença 
O licenciante concede ao licenciado uma licença mundial, gratuita, não exclusiva e sublicenciável, pelo período de duração dos direitos de autor sobre a obra original, para praticar os seguintes atos: 
—  utilizar a obra em quaisquer circunstâncias e para quaisquer fins, 
—  reproduzir a obra, 
—  modificar a obra e fazer obras derivadas baseadas na obra, 
—  comunicar a obra ao público, nomeadamente disponibilizando ou apresentando a obra ou cópias da obra ao público ou executando-a publicamente, consoante o caso,
—  distribuir a obra ou cópias desta, 
—  emprestar e locar a obra ou cópias desta, 
—  sublicenciar os direitos sobre a obra ou cópias desta. 
Estes direitos podem ser exercidos através de quaisquer meios, suportes ou formatos, conhecidos ou que venham a ser inventados, na medida em que tal seja permitido nos termos da legislação aplicável. Nos países em que sejam reconhecidos direitos morais, o licenciante renuncia aos seus direitos morais na medida em que a lei o permita, a fim de tornar efetiva a licença dos direitos patrimoniais acima enumerados. O licenciante concede ao licenciado, a título gratuito e não exclusivo, os direitos de utilização de quaisquer patentes de que o licenciante seja titular, na medida do necessário para a utilização dos direitos concedidos sobre a obra pela presente licença. 

3.Comunicação do código-fonte 
O licenciante pode fornecer a obra como código-fonte ou como código executável. Caso a obra seja fornecida como código executável, o licenciante fornece ainda uma cópia legível por computador do código-fonte da obra, juntamente com cada cópia da obra que o licenciante distribui, ou indica, em aviso colocado a seguir à informação sobre os direitos de autor que acompanha a obra, um repositório onde o código-fonte esteja acessível fácil e gratuitamente enquanto o licenciante continuar a distribuir ou comunicar a obra. 

4.Limitação dos direitos de autor 
A presente licença não pretende, de modo algum, privar o licenciado dos benefícios decorrentes de qualquer exceção ou limitação dos direitos exclusivos dos titulares dos direitos sobre a obra, da caducidade desses direitos ou de outras limitações aplicáveis. 

5.Obrigações previstas na licença 
A concessão dos direitos acima mencionados está sujeita a algumas restrições e obrigações impostas ao licenciado. As obrigações são as seguintes: 

Direito de atribuição: o licenciado deve manter intactos todos os avisos sobre direitos de autor, patentes ou marcas, bem como todos os avisos que se referem à licença e à exclusão de garantia. O licenciado deve juntar uma cópia destes avisos e uma cópia da licença a cada cópia da obra que distribua ou comunique. O licenciado assegura que qualquer obra derivada contenha um aviso bem visível da alteração da obra e da data dessa alteração. 

Cláusula Copyleft: caso o licenciado distribua ou comunique cópias da obra original ou de obras derivadas, tal distribuição ou comunicação é feita nos termos da presente licença ou de uma versão posterior da presente licença, a menos que a obra original seja expressa e exclusivamente distribuída nos termos da presente versão da licença — por exemplo, através da comunicação «Distribuída apenas nos termos da licença EUPL v. 1.2». O licenciado (que se torna licenciante) não pode oferecer ou impor à obra ou à obra derivada termos ou condições suplementares que alterem ou restrinjam os termos da licença. 

Cláusula de compatibilidade: caso o licenciado distribua ou comunique obras derivadas ou cópias destas baseadas na obra e noutras obras licenciadas nos termos de uma licença compatível, tal distribuição ou comunicação pode ser feita nos termos dessa licença compatível. No âmbito da presente cláusula, entende-se por «licença compatível» uma licença enumerada no apêndice anexo à presente licença. Caso as obrigações do licenciado nos termos da licença compatível colidam com as suas obrigações nos termos da presente licença, prevalecem as obrigações da licença compatível. 

Fornecimento do código-fonte: ao distribuir ou comunicar cópias da obra, o licenciado fornece uma cópia legível por computador do código-fonte ou indica um repositório onde este código esteja disponível fácil e gratuitamente enquanto o licenciado continuar a distribuir ou comunicar a obra. 

Proteção jurídica: a presente licença não permite a utilização de firmas, marcas ou nomes do licenciante, exceto na medida do necessário para uma utilização razoável e conforme com as práticas habituais na descrição da origem da obra e na reprodução do teor do aviso sobre os direitos de autor.

6.Cadeia de criação 
O licenciante original garante que os direitos de autor sobre a obra original concedidos pela presente licença lhe pertencem ou lhe foram licenciados e que tem legitimidade para conceder a licença. Cada contribuidor garante que os direitos de autor sobre as alterações que introduziu na obra lhe pertencem ou lhe foram licenciados e que tem legitimidade para conceder a licença. Sempre que o licenciado aceite a licença, o licenciante original e os contribuidores subsequentes concedem ao licenciado uma licença respeitante aos seus contributos para a obra nos termos da presente licença. 

7.Exclusão de garantia 
A obra é um trabalho em curso, continuamente melhorado por numerosos contribuidores. Não é um trabalho acabado, pelo que pode conter defeitos ou erros inerentes a este tipo de trabalho de criação. Por essa razão, a obra é fornecida «tal e qual» nos termos da licença e sem quaisquer garantias, como, entre outras, as respeitantes à comercialização, adequação a um determinado fim, ausência de defeitos ou erros, fiabilidade e não--violação dos direitos de propriedade intelectual distintos dos direitos de autor a que se refere o ponto 6 da presente licença. A presente exclusão de garantia constitui um elemento essencial da licença e uma condição para a concessão de quaisquer direitos sobre a obra. 

8.Exclusão de responsabilidade 
Excetuados os casos de conduta dolosa ou de danos diretamente causados a pessoas singulares, o licenciante não é, em caso algum, responsável por quaisquer danos diretos ou indiretos, materiais ou morais, de qualquer natureza, decorrentes da licença ou da utilização da obra, como, por exemplo, danos por perda de clientela, paragem do trabalho, falhas ou anomalias no funcionamento do equipamento informático, perda de dados ou qualquer dano comercial, ainda que o licenciante tenha sido avisado da possibilidade da ocorrência de tais danos. No entanto, o licenciante é responsável nos termos da legislação relativa à responsabilidade por produtos na medida em que tal legislação seja aplicável à obra. 

9.Acordos suplementares 
Ao distribuir a obra, o licenciado (enquanto licenciante) pode concluir um acordo suplementar, definindo obrigações ou serviços compatíveis com a presente licença. No entanto, ao aceitar essas obrigações, o licenciado (enquanto licenciante) age apenas em seu nome e por sua exclusiva responsabilidade e não em nome do licenciante original ou de qualquer outro contribuidor e apenas no caso de o licenciado (enquanto licenciante) aceitar indemnizar e defender estes contribuidores, exonerando-os de qualquer responsabilidade caso venham a ser acusados como consequência de o licenciado (enquanto licenciante) ter aceitado obrigações de garantia ou de responsabilidade suplementar. 

10.Aceitação da licença 
Para aceitar o disposto na presente licença, o licenciado pode clicar no ícone «Aceito», situado na parte inferior de uma janela que contém o texto da presente licença, ou exprimir o seu consentimento por qualquer outra forma similar, em conformidade com a legislação aplicável. Ao clicar no referido ícone, o licenciado aceita de forma clara e irrevogável a presente licença e os seus termos e condições. Do mesmo modo, o licenciado aceita irrevogavelmente a presente licença, bem como os seus termos e condições, ao exercer quaisquer direitos concedidos pelo ponto 2 da presente licença, nomeadamente a utilização da obra, a criação pelo licenciado de uma obra derivada ou a distribuição ou comunicação pelo licenciado da obra ou de cópias desta. 

11.Informação ao público 
Caso o licenciado distribua ou comunique a obra através de meios de comunicação eletrónicos (por exemplo, oferecendo a possibilidade de a telecarregar), o canal ou meio de distribuição (por exemplo, um sítio web) deve fornecer ao público, no mínimo, a informação exigida pela legislação aplicável no que respeita ao licenciante, à licença e ao modo como pode ser obtida, concluída, armazenada e reproduzida pelo licenciado.

12.Cessação da licença 
A licença e os direitos por ela concedidos cessam automaticamente caso o licenciado não respeite o disposto na licença. Esta cessação não acarreta a cessação das licenças das pessoas que tenham recebido a obra do licenciado nos termos da licença, desde que essas pessoas continuem a respeitar plenamente o disposto na licença. 

13.Diversos 
Sem prejuízo do disposto no ponto 9, a licença representa o acordo integral entre as Partes no que respeita à obra. Caso uma disposição da licença seja inválida ou ineficaz por força da legislação aplicável, tal não afeta a validade ou a eficácia da licença no seu todo. Tal disposição será interpretada ou reformulada na medida do necessário para a tornar válida e eficaz. A Comissão Europeia pode publicar outras versões linguísticas, novas versões da presente licença ou versões atualizadas do apêndice, na medida em que tal seja necessário e razoável, sem reduzir o âmbito de aplicação dos direitos concedidos pela licença. As novas versões da licença serão publicadas com um número de versão único. Todas as versões linguísticas da presente licença, aprovadas pela Comissão Europeia, têm idêntico valor. As Partes podem utilizar a versão linguística da sua preferência. 

14.Jurisdição 
Sem prejuízo de um acordo específico entre as Partes, 
—  o Tribunal de Justiça da União Europeia tem competência exclusiva para dirimir qualquer litígio resultante da interpretação da presente licença entre as instituições, órgãos, organismos ou agências da União Europeia enquanto licenciantes, e qualquer licenciado, em conformidade com o disposto no artigo 272.o do Tratado sobre o Funcionamento da União Europeia,
—  o tribunal da residência do licenciante ou do local onde este exerce a sua atividade principal tem competência exclusiva para dirimir qualquer litígio entre outras Partes resultante da interpretação da presente licença. 

15.Direito aplicável 
Sem prejuízo de um acordo específico entre as Partes, 
—  a presente licença é regida pelo direito do Estado-Membro da União Europeia onde o licenciante tenha a sua sede, domicílio ou estabelecimento, 
—  a presente licença é regida pelo direito belga, caso o licenciante não tenha a sua sede, domicílio ou estabelecimento num Estado-Membro da União Europeia.


Apêndice
As «licenças compatíveis» a que se refere o ponto 5 da EUPL são: 
—  GNU General Public License (GPL) v. 2, v. 3 
—  GNU Affero General Public License (AGPL) v. 3 
—  Open Software License (OSL) v. 2.1, v. 3.0 
—  Eclipse Public License (EPL) v. 1.0 
—  CeCILL v. 2.0, v. 2.1 
—  Mozilla Public Licence (MPL) v. 2 
—  GNU Lesser General Public Licence (LGPL) v. 2.1, v. 3 
—  Creative Commons Attribution-ShareAlike v. 3.0 Unported (CC BY-SA 3.0) para obras que não sejam software 
—  European Union Public Licence (EUPL) v. 1.1, v. 1.2 
—  Québec Free and Open-Source Licence — Reciprocity (LiLiQ-R) or Strong Reciprocity (LiLiQ-R+).
A Comissão pode atualizar o presente apêndice a fim de incluir novas versões das licenças supra, sem a produção de uma nova versão da EUPL, desde que tais versões concedam os direitos conferidos pelo ponto 2 da presente licença e protejam o código-fonte abrangido de apropriação exclusiva.
Todas as outras alterações ou aditamentos ao presente apêndice exigem a produção de uma nova versão da EUPL.