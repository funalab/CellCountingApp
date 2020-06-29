// 出版形式を追加するとこ
$(function(){
    var idNom = 1;
    // 追加ボタン押下時イベント
    $('button#add_rate_in_rightsandcps').on('click',function(){
      var process_btn = $('add_rate_in_rightsandcps').val();
      // $('div#temp_rate_in_rightsandcps')
			$("#temp_rate_in_rightsandcps").load("../../templates/temp.html")
      // コピー処理
      .clone(true)
      // 不要なID削除
      .removeAttr("id")
      // 非表示解除
      .removeClass("notDisp")
      // テキストボックスのID追加
      .find("input")
      .attr("id", "rate_in_rightsandcps" + idNom)
      .end()
			.find("div")
			.attr("id","displayarea_rate_in_cps"+ idNom)
			.end()
      // 追加処理
      .appendTo("div#displayarea_rate_in_rightsandcps");
      // ID番号加算
      idNom++;
    });

    // 削除ボタン押下時イベント
    $('button[name=removeButton]').on('click',function(){
        $(this).parent('div').remove();
    });
});

$(function(){
    var idNo = 1;
    // 追加ボタン押下時イベント
    $('button#add_rate_in_cps').on('click',function(){
      var process_btn = $('add_rate_in_cps').val();
      // $('div#temp_rate_in_cps')
			$("#temp_rate_in_cps").load("../../templates/temp.html")
          // コピー処理
          .clone(true)
          // 不要なID削除
          .removeAttr("id")
          // 非表示解除
          .removeClass("notDisp")
          // テキストボックスのID追加
          .find("input")
          .attr("id", "rate_in_cps" + idNo)
          .end()
          // 追加処理
          .appendTo("div#displayarea_rate_in_cps");
      // ID番号加算
      idNo++;
    });

    // 削除ボタン押下時イベント
    $('button[name=removeButton]').on('click',function(){
        $(this).parent('div').remove();
    });
});
