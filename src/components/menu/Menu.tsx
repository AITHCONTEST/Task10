import Favs from "./Favs";
import styles from "./../styles/menu.module.scss"
import React, { forwardRef } from "react";


interface MenuProps {
  tap: () => void;
  style: React.CSSProperties;
}

const Menu = forwardRef<HTMLDivElement, MenuProps>(({ tap, style }, ref) => {
   

    return(
        <div ref={ref} className={styles.menu}  style={style}>
            <a href="https://cloud.uriit.ru/s/MWo10yvg1a3OmtE" target="_blank">Скачать клавиатуру!</a>
            <Favs tap={tap}/>
        </div>
    );
});

export default Menu;